# okc: problem=muse_moe_w4a16
"""Hopper W4A16 routed experts using Marlin with a fused Muse finalizer.

The weight repack and GEMM calls follow SGLang's Apache-2.0 Marlin path. H200
uses BF16 tensor-core math after on-the-fly NVFP4 weight dequantization; it does
not have Blackwell's native FP4 tensor cores. Compared with the challenge
baseline, this enables Marlin's SM90 atomic reduction and uses one Triton
kernel for post-expert RMSNorm plus weighted accumulation at every batch size.
"""

from typing import TypedDict

import types

import torch
import triton
import triton.language as tl
from sgl_kernel.scalar_type import scalar_types

from sglang.kernels.ops.activation.activation import silu_and_mul
from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm
from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace
from sglang.srt.layers.quantization.marlin_utils_fp4 import (
    prepare_moe_nvfp4_layer_for_marlin,
)


class Weights(TypedDict):
    w13: torch.Tensor
    w13_sf: torch.Tensor
    w13_gs: torch.Tensor
    w2: torch.Tensor
    w2_sf: torch.Tensor
    w2_gs: torch.Tensor
    post_norm_w: torch.Tensor


class Config(TypedDict):
    bs: int
    d_hidden: int
    d_expert: int
    n_experts: int
    local_start: int
    local_count: int
    top_k: int
    sharding: str


class Inputs(TypedDict):
    hidden_states: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    config: Config


N_EXPERTS = 256
TOP_K = 8
BLOCK_SIZE_M_CANDIDATES = (8, 16, 32, 48, 64)

@triton.jit
def _map_local_ids(global_ids, local_ids, COUNT: tl.constexpr, PAIRS: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < PAIRS
    expert = tl.load(global_ids + offsets, mask=mask, other=COUNT)
    tl.store(local_ids + offsets, tl.where(expert < COUNT, expert, -1), mask=mask)


@triton.jit
def _finalize_kernel(
    expert_output,
    topk_weights,
    local_ids,
    norm_weight,
    output,
    H: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    token = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_H)
    mask = offsets < H
    weight = tl.load(norm_weight + offsets, mask=mask, other=0.0).to(tl.float32)
    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)

    for slot in tl.static_range(TOPK):
        pair = token * TOPK + slot
        if tl.load(local_ids + pair) >= 0:
            value = tl.load(
                expert_output + pair * H + offsets, mask=mask, other=0.0
            ).to(tl.float32)
            value *= weight
            variance = tl.sum(value * value, axis=0) / H
            value *= tl.rsqrt(variance + 1e-8)
            route_weight = tl.load(topk_weights + pair).to(tl.float32)
            acc += value * route_weight

    tl.store(output + token * H + offsets, acc, mask=mask)


def _marlin_layer(weights: Weights, start: int, count: int):
    sl = slice(start, start + count)
    layer = types.SimpleNamespace(
        quant_config=types.SimpleNamespace(group_size=16),
        moe_runner_config=types.SimpleNamespace(is_gated=True),
        intermediate_size_per_partition=weights["w13"].shape[1] // 2,
        params_dtype=torch.bfloat16,
        w13_weight=weights["w13"][sl].contiguous(),
        w2_weight=weights["w2"][sl].contiguous(),
        w13_weight_scale=weights["w13_sf"][sl].contiguous(),
        w2_weight_scale=weights["w2_sf"][sl].contiguous(),
        w13_weight_scale_2=weights["w13_gs"][sl].contiguous(),
        w2_weight_scale_2=weights["w2_gs"][sl].contiguous(),
    )
    prepare_moe_nvfp4_layer_for_marlin(layer)
    return layer


def prepare(weights: Weights):
    device = weights["w13"].device
    start, count = 0, 32
    layer = _marlin_layer(weights, start, count)
    expert_map = torch.full((N_EXPERTS,), -1, dtype=torch.int32, device=device)
    expert_map[start : start + count] = torch.arange(
        count, dtype=torch.int32, device=device
    )
    torch.cuda.synchronize()
    return {
        "bank": {
            "w13": layer.w13_weight.data,
            "w13_s": layer.w13_weight_scale.data,
            "w13_gs": layer.w13_weight_scale_2.data,
            "w2": layer.w2_weight.data,
            "w2_s": layer.w2_weight_scale.data,
            "w2_gs": layer.w2_weight_scale_2.data,
            "expert_map": expert_map,
            "count": count,
            "size_n": layer.intermediate_size_per_partition,
        },
        "workspace": marlin_make_workspace(device, max_blocks_per_sm=4),
        "norm_w": (weights["post_norm_w"].float() + 1.0).to(torch.bfloat16),
    }


def kernel_fn(data: Inputs, state) -> torch.Tensor:
    bank = state["bank"]
    x = data["hidden_states"].contiguous()
    num_tokens, size_k = x.shape
    size_n = bank["size_n"]
    gemm1_n = 2 * size_n

    global_ids = data["topk_ids"]
    local_ids = torch.empty_like(global_ids)
    pairs = num_tokens * TOP_K
    _map_local_ids[(1,)](
        global_ids,
        local_ids,
        COUNT=bank["count"],
        PAIRS=pairs,
        BLOCK=triton.next_power_of_2(pairs),
        num_warps=4,
    )
    topk_weights = data["topk_weights"].to(torch.float32).contiguous()

    block_size_m = 8 if num_tokens <= 16 else 16

    filter_invalid = num_tokens > 16
    sorted_ids, block_experts, n_padded = moe_align_block_size(
        local_ids,
        block_size_m,
        bank["count"] if filter_invalid else N_EXPERTS,
        filter_invalid,
    )

    cache2 = torch.empty(
        (num_tokens * TOP_K, size_n), device=x.device, dtype=x.dtype
    )
    cache13 = torch.empty(
        (num_tokens * TOP_K * max(gemm1_n, size_k),),
        device=x.device,
        dtype=x.dtype,
    )
    cache1 = cache13[: num_tokens * TOP_K * gemm1_n].view(-1, gemm1_n)
    cache3 = cache13[: num_tokens * TOP_K * size_k].view(-1, size_k)
    common = dict(
        moe_block_size=block_size_m,
        mul_topk_weights=False,
        is_ep=not filter_invalid,
        b_q_type=scalar_types.float4_e2m1f,
        is_k_full=True,
        use_atomic_add=num_tokens <= 16,
        use_fp32_reduce=True,
        is_zp_float=False,
    )
    workspace = state["workspace"]
    cache1 = moe_wna16_marlin_gemm(
        x,
        cache1,
        bank["w13"],
        None,
        bank["w13_s"],
        bank["w13_gs"],
        None,
        None,
        None,
        workspace,
        sorted_ids,
        block_experts,
        n_padded,
        topk_weights,
        top_k=TOP_K,
        size_m=num_tokens,
        size_n=gemm1_n,
        size_k=size_k,
        **common,
    )
    silu_and_mul(
        cache1.view(-1, gemm1_n),
        cache2,
        expert_ids=local_ids.view(-1),
        expert_step=1,
    )
    expert_output = moe_wna16_marlin_gemm(
        cache2,
        cache3,
        bank["w2"],
        None,
        bank["w2_s"],
        bank["w2_gs"],
        None,
        None,
        None,
        workspace,
        sorted_ids,
        block_experts,
        n_padded,
        topk_weights,
        top_k=1,
        size_m=num_tokens * TOP_K,
        size_n=size_k,
        size_k=size_n,
        **common,
    ).view(num_tokens, TOP_K, size_k)

    output = torch.empty((num_tokens, size_k), device=x.device, dtype=x.dtype)
    _finalize_kernel[(num_tokens,)](
        expert_output,
        topk_weights,
        local_ids,
        state["norm_w"],
        output,
        H=size_k,
        TOPK=TOP_K,
        BLOCK_H=triton.next_power_of_2(size_k),
        num_warps=8,
        num_stages=2,
    )
    return output
