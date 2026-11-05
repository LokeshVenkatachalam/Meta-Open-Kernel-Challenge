# okc: problem=muse_bf16_proj
"""SGLang BF16 projections with a fused split-K router."""

from typing import TypedDict

import torch
import triton
import triton.language as tl
from sglang.kernels.ops.gemm.hopper_bf16_gemv import hopper_bf16_gemv
from sglang.srt.layers.quantization.unquant import _bf16_gemm_dispatch_impl


class Weights(TypedDict):
    qkv_gate: torch.Tensor  # bfloat16 [2304, 8192]
    o: torch.Tensor         # bfloat16 [8192, 1024]
    pooler: torch.Tensor    # bfloat16 [4096, 8192]
    unpooler: torch.Tensor  # bfloat16 [8192, 4096]
    router: torch.Tensor    # float32 [256, 8192]


class Config(TypedDict):
    m: int
    n: int
    k: int
    layer: str


class Inputs(TypedDict):
    x: torch.Tensor
    config: Config


@triton.jit
def _dense_parts(
    x_ptr,
    w_ptr,
    parts_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    split = pid % SPLIT_K
    pid_n = pid // SPLIT_K
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    split_k = K // SPLIT_K
    k_start = split * split_k
    x_ptrs = x_ptr + offs_m[:, None] * K + k_start + offs_k[None, :]
    w_ptrs = w_ptr + offs_n[:, None] * K + k_start + offs_k[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for _ in range(0, split_k, BLOCK_K):
        x = tl.load(x_ptrs, mask=offs_m[:, None] < M, other=0.0)
        w = tl.load(w_ptrs, mask=offs_n[:, None] < N, other=0.0)
        acc += tl.dot(x, tl.trans(w))
        x_ptrs += BLOCK_K
        w_ptrs += BLOCK_K

    offsets = split * M * N + offs_m[:, None] * N + offs_n[None, :]
    tl.store(
        parts_ptr + offsets,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.jit
def _router_parts(
    x_ptr,
    hi_ptr,
    lo_ptr,
    parts_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid = tl.program_id(0)
    split = pid % SPLIT_K
    tile = pid // SPLIT_K
    pid_m = tile % num_pid_m
    pid_n = tile // num_pid_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    split_k = K // SPLIT_K
    k_start = split * split_k
    x_ptrs = x_ptr + offs_m[:, None] * K + k_start + offs_k[None, :]
    hi_ptrs = hi_ptr + offs_n[:, None] * K + k_start + offs_k[None, :]
    lo_ptrs = lo_ptr + offs_n[:, None] * K + k_start + offs_k[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for _ in range(0, split_k, BLOCK_K):
        x = tl.load(x_ptrs, mask=offs_m[:, None] < M, other=0.0)
        hi = tl.load(hi_ptrs, mask=offs_n[:, None] < N, other=0.0)
        lo = tl.load(lo_ptrs, mask=offs_n[:, None] < N, other=0.0)
        acc += tl.dot(x, tl.trans(hi))
        acc += tl.dot(x, tl.trans(lo))
        x_ptrs += BLOCK_K
        hi_ptrs += BLOCK_K
        lo_ptrs += BLOCK_K

    offsets = split * M * N + offs_m[:, None] * N + offs_n[None, :]
    tl.store(
        parts_ptr + offsets,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.jit
def _sum_parts(
    parts_ptr,
    out_ptr,
    ELEMENTS: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < ELEMENTS
    total = tl.zeros((BLOCK,), tl.float32)
    for split in range(0, SPLIT_K):
        total += tl.load(
            parts_ptr + split * ELEMENTS + offsets,
            mask=mask,
            other=0.0,
        )
    tl.store(out_ptr + offsets, total, mask=mask)


def prepare(weights: Weights):
    state = {name: weight.contiguous() for name, weight in weights.items()}
    router = state["router"]
    hi = router.to(torch.bfloat16)
    state["router_hi"] = hi
    state["router_lo"] = (router - hi.float()).to(torch.bfloat16)
    state["router_parts"] = torch.empty(2097152, device=router.device, dtype=torch.float32)
    state["dense_parts"] = torch.empty(2097152, device=router.device, dtype=torch.float32)
    return state


def kernel_fn(data: Inputs, state) -> torch.Tensor:
    x = data["x"]
    config = data["config"]
    m, n, k = config["m"], config["n"], config["k"]
    layer = config["layer"]

    if layer != "router":
        weight = state[layer][:n]
        if m == 1 and layer != "unpooler":
            return hopper_bf16_gemv(x, weight)

        use_split = (
            (layer == "pooler" and m <= 8)
            or (layer == "unpooler" and n == 1024)
        )
        if not use_split:
            return _bf16_gemm_dispatch_impl(x, weight, None)

        split_k = 32
        out = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
        _dense_parts[(triton.cdiv(n, 128) * split_k,)](
            x,
            weight,
            state["dense_parts"],
            M=m,
            N=n,
            K=k,
            SPLIT_K=split_k,
            BLOCK_M=16,
            BLOCK_N=128,
            BLOCK_K=64,
            num_warps=8,
            num_stages=3,
        )
        elements = m * n
        reduce_block = 1024 if m == 16 else 256
        reduce_warps = 8 if m == 16 else 4
        _sum_parts[(triton.cdiv(elements, reduce_block),)](
            state["dense_parts"],
            out,
            ELEMENTS=elements,
            SPLIT_K=split_k,
            BLOCK=reduce_block,
            num_warps=reduce_warps,
        )
        return out

    block_m = 16 if m <= 16 else min(m, 128)
    split_k = 32
    num_warps = 8
    out = torch.empty((m, n), device=x.device, dtype=torch.float32)
    grid = (
        triton.cdiv(m, block_m) * triton.cdiv(n, 64) * split_k,
    )
    _router_parts[grid](
        x,
        state["router_hi"],
        state["router_lo"],
        state["router_parts"],
        M=m,
        N=n,
        K=k,
        SPLIT_K=split_k,
        BLOCK_M=block_m,
        BLOCK_N=64,
        BLOCK_K=64,
        num_warps=num_warps,
        num_stages=2,
    )
    elements = m * n
    _sum_parts[(triton.cdiv(elements, 256),)](
        state["router_parts"],
        out,
        ELEMENTS=elements,
        SPLIT_K=split_k,
        BLOCK=256,
        num_warps=4,
    )
    return out
