# okc: problem=muse_attention_decode
"""Fused Muse Spark attention preparation around the exact FA3 paged-attention call.

The Triton kernel combines Q/K RMSNorm, optional interleaved RoPE, FP8 query
conversion, and scaled K/V cache writes. FA3 still attends every visible key;
the final per-head RMSNorm and sigmoid gate use SGLang's fused output kernel.
"""

from typing import TypedDict

import torch
import triton
import triton.language as tl
from sgl_kernel.flash_attn import flash_attn_with_kvcache
from sglang.kernels.ops.muse_spark_v12.norm_gate import rms_norm_gate


class Config(TypedDict):
    batch: int
    q_len: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    page_size: int
    window: int
    rope_theta: float
    qk_scale_factor: float
    sm_scale: float
    kv_scale: float


class Inputs(TypedDict):
    qkv: torch.Tensor
    positions: torch.Tensor
    cache_loc: torch.Tensor
    cache_seqlens: torch.Tensor
    cu_seqlens_q: torch.Tensor
    scheduler_metadata: torch.Tensor | None
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    k_scale: torch.Tensor
    v_scale: torch.Tensor
    page_table: torch.Tensor
    seq_lens: torch.Tensor
    config: Config


RMS_EPS = 1e-5
FP8 = torch.float8_e4m3fn
FP8_MAX = 448.0
ROPE_POSITIONS = 1 << 21
_ROPE = {}


def _cos_sin(theta: float, d: int, device: torch.device) -> torch.Tensor:
    key = (theta, d, device)
    cache = _ROPE.get(key)
    if cache is None:
        inv_freq = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float32, device=device) / d))
        positions = torch.arange(ROPE_POSITIONS, dtype=torch.float32, device=device)
        freqs = torch.einsum("i,j->ij", positions, inv_freq)
        cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1)
        _ROPE[key] = cache
    return cache


@triton.jit
def _prepare_qkv(
    qkv,
    q_out,
    positions,
    cos_sin,
    k_cache,
    v_cache,
    cache_loc,
    k_scale,
    v_scale,
    stride_qkv,
    n_rows,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    HAS_ROPE: tl.constexpr,
    SCALED_KV: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < n_rows
    cols = tl.arange(0, D)
    mask = row_mask[:, None]

    heads_per_token: tl.constexpr = NUM_Q_HEADS + 2 * NUM_KV_HEADS
    token = rows // heads_per_token
    head = rows % heads_per_token
    is_q = head < NUM_Q_HEADS
    is_k = (head >= NUM_Q_HEADS) & (head < NUM_Q_HEADS + NUM_KV_HEADS)
    is_v = head >= NUM_Q_HEADS + NUM_KV_HEADS

    src = token * stride_qkv + head * D
    raw = tl.load(qkv + src[:, None] + cols[None, :], mask=mask, other=0.0).to(tl.float32)
    mean_square = tl.sum(raw * raw, axis=1) / D
    rstd = tl.rsqrt(mean_square + 1e-5)
    normed = raw * rstd[:, None]

    if HAS_ROPE:
        partner = tl.load(
            qkv + src[:, None] + (cols ^ 1)[None, :], mask=mask, other=0.0
        ).to(tl.float32)
        partner = partner * rstd[:, None]
        pos = tl.load(positions + token, mask=row_mask, other=0).to(tl.int64)
        freq = cols // 2
        cos = tl.load(cos_sin + pos[:, None] * D + freq[None, :], mask=mask, other=1.0)
        sin = tl.load(
            cos_sin + pos[:, None] * D + D // 2 + freq[None, :], mask=mask, other=0.0
        )
        even = (cols % 2 == 0)[None, :]
        normed = tl.where(even, normed * cos - partner * sin, partner * sin + normed * cos)

    value = tl.where(is_v[:, None], raw, normed)
    if SCALED_KV:
        ks = tl.load(k_scale)
        vs = tl.load(v_scale)
        value = tl.where(is_k[:, None], value / ks, value)
        value = tl.where(is_v[:, None], value / vs, value)
    value = tl.maximum(-448.0, tl.minimum(448.0, value))

    q_dst = (token * NUM_Q_HEADS + head) * D
    tl.store(q_out + q_dst[:, None] + cols[None, :], value, mask=mask & is_q[:, None])

    loc = tl.load(cache_loc + token, mask=row_mask, other=0).to(tl.int64)
    kv_head = tl.where(is_v, head - NUM_Q_HEADS - NUM_KV_HEADS, head - NUM_Q_HEADS)
    kv_dst = (loc * NUM_KV_HEADS + kv_head) * D
    tl.store(k_cache + kv_dst[:, None] + cols[None, :], value, mask=mask & is_k[:, None])
    tl.store(v_cache + kv_dst[:, None] + cols[None, :], value, mask=mask & is_v[:, None])


def _prepare(data: Inputs, qkv: torch.Tensor, tokens: int, hq: int, hkv: int, d: int) -> torch.Tensor:
    q = torch.empty((tokens, hq, d), dtype=FP8, device=qkv.device)
    n_rows = tokens * (hq + 2 * hkv)
    block_m = 4 if n_rows < 1024 else 8
    has_rope = bool(data["config"]["rope_theta"])
    cos_sin = _cos_sin(data["config"]["rope_theta"], d, qkv.device) if has_rope else qkv
    _prepare_qkv[(triton.cdiv(n_rows, block_m),)](
        qkv,
        q,
        data["positions"],
        cos_sin,
        data["k_cache"],
        data["v_cache"],
        data["cache_loc"],
        data["k_scale"],
        data["v_scale"],
        qkv.stride(0),
        n_rows,
        NUM_Q_HEADS=hq,
        NUM_KV_HEADS=hkv,
        D=d,
        BLOCK_M=block_m,
        HAS_ROPE=has_rope,
        SCALED_KV=data["config"]["kv_scale"] != 1.0,
        num_warps=4,
    )
    return q


def kernel_fn(data: Inputs) -> torch.Tensor:
    cfg = data["config"]
    batch, length = cfg["batch"], cfg["q_len"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    tokens = batch * length
    qkv = data["qkv"].view(tokens, -1)
    q = _prepare(data, qkv, tokens, hq, hkv, d)

    descale = {}
    if cfg["kv_scale"] != 1.0:
        descale = {
            "k_descale": data["k_scale"].reshape(1, 1).expand(batch, hkv),
            "v_descale": data["v_scale"].reshape(1, 1).expand(batch, hkv),
        }
    window = cfg["window"]
    out = flash_attn_with_kvcache(
        q=q.view(batch, length, hq, d),
        k_cache=data["k_cache"],
        v_cache=data["v_cache"],
        page_table=data["page_table"],
        cache_seqlens=data["cache_seqlens"],
        max_seqlen_q=length,
        softmax_scale=cfg["sm_scale"] * cfg["qk_scale_factor"],
        causal=True,
        window_size=(window - 1, 0) if window else (-1, -1),
        scheduler_metadata=data["scheduler_metadata"],
        **descale,
    )
    if isinstance(out, tuple):
        out = out[0]
    gate = qkv[:, (hq + 2 * hkv) * d :]
    out = rms_norm_gate(
        out.reshape(tokens, hq * d).to(torch.bfloat16),
        gate,
        None,
        head_dim=d,
        eps=RMS_EPS,
        has_norm=True,
    )
    return out.view(batch, length, hq * d)
