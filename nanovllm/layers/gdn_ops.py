"""Triton kernels for Qwen3.5's Gated Delta Net (linear attention).

Ported from vLLM 0.26.0's vendored flash-linear-attention kernels; the deliberate
differences from upstream (no autotune, the dropped TMA/ROCm/prefix-caching branches, the
re-implemented conv kernels) are recorded in Devlog/v0.3.1.md item 23.

Layout conventions, identical to vLLM/FLA -- and what ``MambaShape`` in
``nanovllm/engine/block_manager.py`` assumes:

    q/k/v            [1, T, H, D]                (varlen: sequences packed along T)
    g / beta         [1, T, H]                   g is log-decay (<= 0)
    ssm (temporal)   [num_slots, HV, V, K]       fp32, K innermost
    conv             [num_slots, conv_dim, W-1]
    conv input       [T, conv_dim]

Head layout follows GQA: ``H`` value heads share ``Hg`` key heads, value head ``h`` uses
key head ``h // (H // Hg)``. The kernels index that directly, so callers must *not*
``repeat_interleave`` q/k the way HF's reference implementation does.
"""

import torch
import triton
import triton.language as tl

FLA_CHUNK_SIZE = 64
L2NORM_EPS = 1e-6
SOFTPLUS_THRESHOLD = 20.0


# ---------------------------------------------------------------------------
# index helpers -- tensor ops, so only the grid size (a host int) forces one
# device->host read per prefill step
# ---------------------------------------------------------------------------
def prepare_chunk_indices(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """[NT, 2] int32 mapping global chunk slot -> (sequence, chunk within sequence)."""
    lens = cu_seqlens[1:] - cu_seqlens[:-1]
    nt = (lens + chunk_size - 1) // chunk_size
    n_chunks = nt.sum().item()
    offsets = torch.cumsum(nt, 0) - nt
    seq = torch.repeat_interleave(torch.arange(nt.numel(), device=nt.device), nt)
    loc = torch.arange(n_chunks, device=nt.device, dtype=nt.dtype) - torch.repeat_interleave(offsets, nt)
    return torch.stack([seq, loc], 1).to(torch.int32)


def prepare_chunk_offsets(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """[N+1] int32: first chunk slot of each sequence (prefix sum of chunk counts)."""
    lens = cu_seqlens[1:] - cu_seqlens[:-1]
    nt = (lens + chunk_size - 1) // chunk_size
    return torch.cat([nt.new_zeros(1), torch.cumsum(nt, 0)]).to(torch.int32)


# ---------------------------------------------------------------------------
# l2norm (FLA ops/l2norm.py: l2norm_fwd_kernel2)
# ---------------------------------------------------------------------------
@triton.jit
def _l2norm_fwd_kernel(
    X, Y, eps, M, N: tl.constexpr, BD: tl.constexpr, MBLOCK: tl.constexpr
):
    xoffset = tl.program_id(0) * MBLOCK
    row_idx = xoffset + tl.arange(0, MBLOCK)[:, None]
    xmask = row_idx < M
    rindex = tl.arange(0, BD)[None, :]
    cmask = rindex < N
    mask = xmask & cmask
    xs = tl.load(X + (rindex + N * row_idx), mask, other=0.0).to(tl.float32)
    square = tl.broadcast_to(xs * xs, [MBLOCK, BD])
    square_sum = tl.sum(tl.where(xmask, square, 0), 1)[:, None]
    rsqrt = tl.rsqrt(square_sum + eps)
    tl.store(Y + (rindex + N * row_idx), xs * rsqrt, mask)


def l2norm_fwd(x: torch.Tensor, eps: float = L2NORM_EPS) -> torch.Tensor:
    """Row-wise L2 normalization, accumulated in fp32 (matches FLA's kernel)."""
    shape_og = x.shape
    x = x.reshape(-1, x.shape[-1])
    if not x.is_contiguous():
        x = x.contiguous()
    y = torch.empty_like(x)
    T, D = x.shape
    BD = min(65536 // x.element_size(), triton.next_power_of_2(D))
    assert D <= BD, "l2norm: feature dim >= 64KB is not supported"
    MBLOCK = 32
    _l2norm_fwd_kernel[(triton.cdiv(T, MBLOCK),)](x, y, eps, T, D, BD, MBLOCK)
    return y.view(shape_og)


# ---------------------------------------------------------------------------
# causal depthwise conv1d over the packed [q, k, v] projection (silu applied)
#
# A state slot holds the last W-1 *raw* inputs: ``nn.Conv1d`` with left padding,
# plus a persistent window that a continuing chunk reads back and overwrites
# ---------------------------------------------------------------------------
@triton.jit(do_not_specialize=["T"])
def _causal_conv1d_prefill_kernel(
    x,                       # [T, D], channel-minor
    weight,                  # [D, W]
    bias,                    # [D] or dummy
    out,                     # [T, D]
    conv_states,             # [num_slots, D, W-1]
    cu_seqlens,
    chunk_indices,
    in_idx,                  # [N] int32, -1 => no initial window
    out_idx,                 # [N] int32
    T,
    D: tl.constexpr,
    W: tl.constexpr,
    S2: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    IS_BIAS: tl.constexpr,
):
    i_t, i_d = tl.program_id(0), tl.program_id(1)
    i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
    i_c = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
    bos = tl.load(cu_seqlens + i_n).to(tl.int32)
    eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    T = eos - bos
    S: tl.constexpr = W - 1

    # tiles are [BT, BD] so that the last axis is the (contiguous) channel axis
    o_t = i_c * BT + tl.arange(0, BT)
    m_t = o_t < T
    o_d = i_d * BD + tl.arange(0, BD)
    m_d = o_d < D

    slot_in = tl.load(in_idx + i_n).to(tl.int64)
    has_state = slot_in >= 0
    p_st_in = conv_states + slot_in * D * S

    b_acc = tl.zeros([BT, BD], dtype=tl.float32)
    for j in tl.static_range(W):
        b_wj = tl.load(weight + o_d * W + j, mask=m_d, other=0.0)
        row = o_t + (j - S)
        m = (row >= 0) & (row < T) & m_t
        b_x = tl.load(
            x + (bos + row)[:, None] * D + o_d[None, :],
            mask=m[:, None] & m_d[None, :],
            other=0.0,
        )
        b_acc += b_x.to(tl.float32) * b_wj[None, :]

    # read and write-back of the window stay in one program (the first chunk's), which
    # rules out a cross-program race when in_idx == out_idx -- a chunked-prefill
    # continuation re-reads and overwrites its own slot
    if i_c == 0:
        if has_state:
            for j in tl.static_range(W - 1):
                b_wj = tl.load(weight + o_d * W + j, mask=m_d, other=0.0)
                row = o_t + (j - S)
                m = (row < 0) & m_t
                # conv state is [D, W-1]: the channel axis carries stride W-1
                b_st = tl.load(
                    p_st_in + o_d[None, :] * S + (S + row)[:, None],
                    mask=m[:, None] & m_d[None, :],
                    other=0.0,
                )
                b_acc += b_st.to(tl.float32) * b_wj[None, :]

        # new window = trailing S raw inputs (shorter sequences: what there is of them)
        slot_out = tl.load(out_idx + i_n).to(tl.int64)
        if slot_out >= 0:
            p_st_out = conv_states + slot_out * D * S
            o_s = tl.arange(0, S2)
            col = T - S + o_s
            m_col = (o_s < S) & (col >= 0)
            b_win = tl.load(
                x + (bos + col)[:, None] * D + o_d[None, :],
                mask=m_col[:, None] & m_d[None, :],
                other=0.0,
            )
            if has_state:
                m_col = (o_s < S) & (col < 0)
                b_win += tl.load(
                    p_st_in + (S + col)[:, None] + o_d[None, :] * S,
                    mask=m_col[:, None] & m_d[None, :],
                    other=0.0,
                )
            tl.debug_barrier()
            tl.store(
                p_st_out + o_s[:, None] + o_d[None, :] * S,
                b_win,
                mask=(o_s < S)[:, None] & m_d[None, :],
            )

    if IS_BIAS:
        b_acc += tl.load(bias + o_d, mask=m_d, other=0.0).to(tl.float32)[None, :]
    b_acc = b_acc / (1 + tl.exp(-b_acc))  # silu
    tl.store(
        out + (bos + o_t)[:, None] * D + o_d[None, :],
        b_acc.to(out.dtype.element_ty),
        mask=m_t[:, None] & m_d[None, :],
    )


def causal_conv1d_prefill(
    x: torch.Tensor,            # [T, D] raw packed qkv
    weight: torch.Tensor,       # [D, W]
    bias: torch.Tensor | None,
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
    conv_state: torch.Tensor,   # [num_slots, D, W-1]
    in_idx: torch.Tensor,       # [N] int32, -1 for "no window"
    out_idx: torch.Tensor,      # [N] int32
) -> torch.Tensor:
    T, D = x.shape
    W = weight.shape[-1]
    assert W >= 2, "a depthwise conv with kernel size 1 has no state window"
    if not x.is_contiguous():
        x = x.contiguous()
    if not weight.is_contiguous():
        weight = weight.contiguous()
    out = torch.empty_like(x)
    BD = min(triton.next_power_of_2(D), 64)
    grid = (chunk_indices.shape[0], triton.cdiv(D, BD))
    _causal_conv1d_prefill_kernel[grid](
        x,
        weight,
        bias if bias is not None else weight,
        out,
        conv_state,
        cu_seqlens,
        chunk_indices,
        in_idx,
        out_idx,
        T=T,
        D=D,
        W=W,
        S2=triton.next_power_of_2(max(W - 1, 1)),
        BT=FLA_CHUNK_SIZE,
        BD=BD,
        IS_BIAS=bias is not None,
        num_warps=4,
    )
    return out


@triton.jit
def _causal_conv1d_update_kernel(
    x,                       # [B, D]
    weight,                  # [D, W]
    bias,
    out,                     # [B, D]
    conv_states,             # [num_slots, D, W-1]
    state_idx,               # [B] int32
    D: tl.constexpr,
    W: tl.constexpr,
    S2: tl.constexpr,
    BD: tl.constexpr,
    IS_BIAS: tl.constexpr,
):
    i_d, i_n = tl.program_id(0), tl.program_id(1)
    S: tl.constexpr = W - 1
    o_d = i_d * BD + tl.arange(0, BD)
    m_d = o_d < D
    o_s = tl.arange(0, S2)
    m_s = (o_s < S)[None, :]

    slot = tl.load(state_idx + i_n).to(tl.int64)
    if slot < 0:
        # padded row: ModelRunner fills the CUDA-graph batch padding with -1
        tl.store(out + i_n * D + o_d, tl.zeros([BD], dtype=out.dtype.element_ty), mask=m_d)
        return
    p_st = conv_states + slot * D * S
    # read the window and the incoming token before touching the state
    b_win = tl.load(
        p_st + o_d[:, None] * S + o_s[None, :], mask=m_d[:, None] & m_s, other=0.0
    )
    b_w = tl.load(
        weight + o_d[:, None] * W + o_s[None, :], mask=m_d[:, None] & m_s, other=0.0
    )
    b_x = tl.load(x + i_n * D + o_d, mask=m_d, other=0.0)

    b_acc = tl.sum(b_win.to(tl.float32) * b_w.to(tl.float32), axis=1)
    b_acc += b_x.to(tl.float32) * tl.load(weight + o_d * W + (W - 1), mask=m_d, other=0.0)
    if IS_BIAS:
        b_acc += tl.load(bias + o_d, mask=m_d, other=0.0).to(tl.float32)
    b_acc = b_acc / (1 + tl.exp(-b_acc))  # silu
    tl.store(out + i_n * D + o_d, b_acc.to(out.dtype.element_ty), mask=m_d)

    # new window: shift left by one and append the incoming token
    tl.store(
        p_st + o_d[:, None] * S + (o_s[None, :] - 1),
        b_win,
        mask=m_d[:, None] & (o_s >= 1)[None, :] & m_s,
    )
    tl.store(p_st + o_d * S + (S - 1), b_x, mask=m_d)


def causal_conv1d_update(
    x: torch.Tensor,            # [B, D] raw packed qkv of the current token
    weight: torch.Tensor,       # [D, W]
    bias: torch.Tensor | None,
    conv_state: torch.Tensor,   # [num_slots, D, W-1], updated in place
    state_idx: torch.Tensor,    # [B] int32
) -> torch.Tensor:
    B, D = x.shape
    W = weight.shape[-1]
    assert W >= 2, "a depthwise conv with kernel size 1 has no state window"
    if not x.is_contiguous():
        x = x.contiguous()
    if not weight.is_contiguous():
        weight = weight.contiguous()
    out = torch.empty_like(x)
    BD = min(triton.next_power_of_2(D), 256)
    grid = (triton.cdiv(D, BD), B)
    _causal_conv1d_update_kernel[grid](
        x,
        weight,
        bias if bias is not None else weight,
        out,
        conv_state,
        state_idx,
        D=D,
        W=W,
        S2=triton.next_power_of_2(max(W - 1, 1)),
        BD=BD,
        IS_BIAS=bias is not None,
        num_warps=4,
    )
    return out


# ---------------------------------------------------------------------------
# prefill -- chunked gated delta rule, mirroring vLLM's chunk_gated_delta_rule_fwd()
#
#   cumsum(g) -> A = kkt -> Ai = (I + A)^-1 -> (w, u) -> h/v_new -> o
# ---------------------------------------------------------------------------
@triton.jit(do_not_specialize=["T"])
def _chunk_local_cumsum_kernel(
    s,
    o,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
        i_t = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    p_s = tl.make_block_ptr(s + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_o = tl.make_block_ptr(o + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    b_s = tl.load(p_s, boundary_check=(0,)).to(tl.float32)
    b_o = tl.cumsum(b_s, axis=0)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0,))


def chunk_local_cumsum(
    g: torch.Tensor,           # [B, T, H]
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
    chunk_size: int = FLA_CHUNK_SIZE,
) -> torch.Tensor:
    """Cumulative log-decay inside each chunk (resets at every chunk boundary)."""
    if not g.is_contiguous():
        g = g.contiguous()
    B, T, H = g.shape
    o = torch.empty_like(g, dtype=torch.float32)
    BT = chunk_size
    NT = chunk_indices.shape[0] if cu_seqlens is not None else triton.cdiv(T, BT)
    _chunk_local_cumsum_kernel[(NT, B * H)](
        g, o, cu_seqlens, chunk_indices, T=T, H=H, BT=BT,
        IS_VARLEN=cu_seqlens is not None, num_warps=4,
    )
    return o


@triton.jit(do_not_specialize=["T"])
def _chunk_scaled_dot_kkt_fwd_kernel(
    k,
    beta,
    g,
    A,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
        i_t = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T

    p_beta = tl.make_block_ptr(
        beta + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,)
    )
    p_g = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    b_beta = tl.load(p_beta, boundary_check=(0,))
    b_g = tl.load(p_g, boundary_check=(0,))

    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(
            k + (bos * Hg + i_h // (H // Hg)) * K,
            (T, K),
            (Hg * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_kb = b_k * b_beta[:, None]
        b_A += tl.dot(b_kb, tl.trans(b_k).to(b_kb.dtype))

    b_g_diff = b_g[:, None] - b_g[None, :]
    b_A = b_A * tl.exp(b_g_diff)
    m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)
    b_A = tl.where(m_A, b_A, 0)
    p_A = tl.make_block_ptr(
        A + (bos * H + i_h) * BT, (T, BT), (BT * H, 1), (i_t * BT, 0), (BT, BT), (1, 0)
    )
    tl.store(p_A, b_A.to(p_A.dtype.element_ty), boundary_check=(0, 1))


def chunk_scaled_dot_kkt_fwd(
    k: torch.Tensor,           # [B, T, Hg, K]
    beta: torch.Tensor,        # [B, T, H]
    g: torch.Tensor,           # [B, T, H] chunk-local cumsum
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
    chunk_size: int = FLA_CHUNK_SIZE,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """A[i, j] = beta_i * (k_i . k_j) * exp(g_i - g_j) for j < i inside a chunk."""
    if not k.is_contiguous():
        k = k.contiguous()
    if not beta.is_contiguous():
        beta = beta.contiguous()
    B, T, Hg, K = k.shape
    H = beta.shape[-1]
    BT = chunk_size
    A = torch.empty(B, T, H, BT, device=k.device, dtype=output_dtype)
    NT = chunk_indices.shape[0] if cu_seqlens is not None else triton.cdiv(T, BT)
    BK = min(triton.next_power_of_2(K), 64)
    _chunk_scaled_dot_kkt_fwd_kernel[(NT, B * H)](
        k, beta, g, A, cu_seqlens, chunk_indices, T=T, H=H, Hg=Hg, K=K, BT=BT, BK=BK,
        IS_VARLEN=cu_seqlens is not None, num_warps=4, num_stages=3,
    )
    return A


@triton.jit(do_not_specialize=["T"])
def _merge_16x16_to_64x64_inverse_kernel(
    A,
    Ai,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    """(I + A)^-1 for a 64x64 strictly-lower-triangular A, via 16x16 blocks."""
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
        i_t = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    o_i = tl.arange(0, 16)
    m_A = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]
    A += (bos * H + i_h) * BT
    Ai += (bos * H + i_h) * BT

    p_A_11 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT, 0), (16, 16), (1, 0))
    p_A_22 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT + 16, 16), (16, 16), (1, 0))
    p_A_33 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT + 32, 32), (16, 16), (1, 0))
    p_A_44 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT + 48, 48), (16, 16), (1, 0))
    b_Ai_11 = tl.load(p_A_11, boundary_check=(0, 1)).to(tl.float32)
    b_Ai_22 = tl.load(p_A_22, boundary_check=(0, 1)).to(tl.float32)
    b_Ai_33 = tl.load(p_A_33, boundary_check=(0, 1)).to(tl.float32)
    b_Ai_44 = tl.load(p_A_44, boundary_check=(0, 1)).to(tl.float32)

    b_Ai_11 = -tl.where(m_A, b_Ai_11, 0)
    b_Ai_22 = -tl.where(m_A, b_Ai_22, 0)
    b_Ai_33 = -tl.where(m_A, b_Ai_33, 0)
    b_Ai_44 = -tl.where(m_A, b_Ai_44, 0)

    # forward substitution, row by row, inside each 16x16 diagonal block
    for i in range(2, min(16, T - i_t * BT)):
        b_a_11 = -tl.load(A + (i_t * BT + i) * H * BT + o_i)
        b_a_11 += tl.sum(b_a_11[:, None] * b_Ai_11, 0)
        b_Ai_11 = tl.where((o_i == i)[:, None], b_a_11, b_Ai_11)
    for i in range(16 + 2, min(32, T - i_t * BT)):
        b_a_22 = -tl.load(A + (i_t * BT + i) * H * BT + o_i + 16)
        b_a_22 += tl.sum(b_a_22[:, None] * b_Ai_22, 0)
        b_Ai_22 = tl.where((o_i == i - 16)[:, None], b_a_22, b_Ai_22)
    for i in range(32 + 2, min(48, T - i_t * BT)):
        b_a_33 = -tl.load(A + (i_t * BT + i) * H * BT + o_i + 32)
        b_a_33 += tl.sum(b_a_33[:, None] * b_Ai_33, 0)
        b_Ai_33 = tl.where((o_i == i - 32)[:, None], b_a_33, b_Ai_33)
    for i in range(48 + 2, min(64, T - i_t * BT)):
        b_a_44 = -tl.load(A + (i_t * BT + i) * H * BT + o_i + 48)
        b_a_44 += tl.sum(b_a_44[:, None] * b_Ai_44, 0)
        b_Ai_44 = tl.where((o_i == i - 48)[:, None], b_a_44, b_Ai_44)
    b_Ai_11 += m_I
    b_Ai_22 += m_I
    b_Ai_33 += m_I
    b_Ai_44 += m_I

    p_A_21 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT + 16, 0), (16, 16), (1, 0))
    p_A_31 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT + 32, 0), (16, 16), (1, 0))
    p_A_32 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT + 32, 16), (16, 16), (1, 0))
    p_A_41 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT + 48, 0), (16, 16), (1, 0))
    p_A_42 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT + 48, 16), (16, 16), (1, 0))
    p_A_43 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT + 48, 32), (16, 16), (1, 0))
    b_A_21 = tl.load(p_A_21, boundary_check=(0, 1)).to(tl.float32)
    b_A_31 = tl.load(p_A_31, boundary_check=(0, 1)).to(tl.float32)
    b_A_32 = tl.load(p_A_32, boundary_check=(0, 1)).to(tl.float32)
    b_A_41 = tl.load(p_A_41, boundary_check=(0, 1)).to(tl.float32)
    b_A_42 = tl.load(p_A_42, boundary_check=(0, 1)).to(tl.float32)
    b_A_43 = tl.load(p_A_43, boundary_check=(0, 1)).to(tl.float32)

    # off-diagonal blocks: Ai_ij = -Ai_ii @ (sum_k A_ik @ Ai_kj)
    b_Ai_21 = -tl.dot(
        tl.dot(b_Ai_22, b_A_21, input_precision=DOT_PRECISION),
        b_Ai_11,
        input_precision=DOT_PRECISION,
    )
    b_Ai_32 = -tl.dot(
        tl.dot(b_Ai_33, b_A_32, input_precision=DOT_PRECISION),
        b_Ai_22,
        input_precision=DOT_PRECISION,
    )
    b_Ai_43 = -tl.dot(
        tl.dot(b_Ai_44, b_A_43, input_precision=DOT_PRECISION),
        b_Ai_33,
        input_precision=DOT_PRECISION,
    )
    b_Ai_31 = -tl.dot(
        b_Ai_33,
        tl.dot(b_A_31, b_Ai_11, input_precision=DOT_PRECISION)
        + tl.dot(b_A_32, b_Ai_21, input_precision=DOT_PRECISION),
        input_precision=DOT_PRECISION,
    )
    b_Ai_42 = -tl.dot(
        b_Ai_44,
        tl.dot(b_A_42, b_Ai_22, input_precision=DOT_PRECISION)
        + tl.dot(b_A_43, b_Ai_32, input_precision=DOT_PRECISION),
        input_precision=DOT_PRECISION,
    )
    b_Ai_41 = -tl.dot(
        b_Ai_44,
        tl.dot(b_A_41, b_Ai_11, input_precision=DOT_PRECISION)
        + tl.dot(b_A_42, b_Ai_21, input_precision=DOT_PRECISION)
        + tl.dot(b_A_43, b_Ai_31, input_precision=DOT_PRECISION),
        input_precision=DOT_PRECISION,
    )

    p_Ai_11 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT, 0), (16, 16), (1, 0))
    p_Ai_22 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT + 16, 16), (16, 16), (1, 0))
    p_Ai_33 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT + 32, 32), (16, 16), (1, 0))
    p_Ai_44 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT + 48, 48), (16, 16), (1, 0))
    p_Ai_21 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT + 16, 0), (16, 16), (1, 0))
    p_Ai_31 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT + 32, 0), (16, 16), (1, 0))
    p_Ai_32 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT + 32, 16), (16, 16), (1, 0))
    p_Ai_41 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT + 48, 0), (16, 16), (1, 0))
    p_Ai_42 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT + 48, 16), (16, 16), (1, 0))
    p_Ai_43 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT + 48, 32), (16, 16), (1, 0))
    tl.store(p_Ai_11, b_Ai_11.to(p_Ai_11.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_Ai_22, b_Ai_22.to(p_Ai_22.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_Ai_33, b_Ai_33.to(p_Ai_33.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_Ai_44, b_Ai_44.to(p_Ai_44.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_Ai_21, b_Ai_21.to(p_Ai_21.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_Ai_31, b_Ai_31.to(p_Ai_31.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_Ai_32, b_Ai_32.to(p_Ai_32.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_Ai_41, b_Ai_41.to(p_Ai_41.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_Ai_42, b_Ai_42.to(p_Ai_42.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    tl.store(p_Ai_43, b_Ai_43.to(p_Ai_43.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))


def solve_tril(
    A: torch.Tensor,           # [B, T, H, BT] float32, strictly lower triangular
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Returns (I + A)^-1, same shape as A."""
    assert A.shape[-1] == 64, "this port ships FLA's 64x64 (BT=64) merge kernel only"
    output_dtype = A.dtype if output_dtype is None else output_dtype
    B, T, H, BT = A.shape
    NT = chunk_indices.shape[0] if cu_seqlens is not None else triton.cdiv(T, BT)
    Ai = torch.zeros_like(A, dtype=output_dtype)
    _merge_16x16_to_64x64_inverse_kernel[(NT, B * H)](
        A=A,
        Ai=Ai,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        BT=BT,
        IS_VARLEN=cu_seqlens is not None,
        DOT_PRECISION="ieee",
        num_warps=4,
        num_stages=3,
    )
    return Ai


@triton.jit(do_not_specialize=["T"])
def _recompute_w_u_fwd_kernel(
    k,
    v,
    beta,
    w,
    u,
    A,
    g,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
        i_t = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
    p_beta = tl.make_block_ptr(beta + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_g = tl.make_block_ptr(g + (bos * H + i_h), (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_A = tl.make_block_ptr(
        A + (bos * H + i_h) * BT, (T, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0)
    )
    b_beta = tl.load(p_beta, boundary_check=(0,))
    b_A = tl.load(p_A, boundary_check=(0, 1))
    b_g = tl.exp(tl.load(p_g, boundary_check=(0,)))

    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(
            v + (bos * H + i_h) * V, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        p_u = tl.make_block_ptr(
            u + (bos * H + i_h) * V, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_vb = (b_v * b_beta[:, None]).to(b_v.dtype)
        b_u = tl.dot(b_A, b_vb, allow_tf32=False)
        tl.store(p_u, b_u.to(p_u.dtype.element_ty), boundary_check=(0, 1))

    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(
            k + (bos * Hg + i_h // (H // Hg)) * K,
            (T, K),
            (Hg * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        p_w = tl.make_block_ptr(
            w + (bos * H + i_h) * K, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_kb = (b_k * b_beta[:, None] * b_g[:, None]).to(b_k.dtype)
        b_w = tl.dot(b_A, b_kb)
        tl.store(p_w, b_w.to(p_w.dtype.element_ty), boundary_check=(0, 1))


def recompute_w_u_fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g_cumsum: torch.Tensor,
    A: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
    chunk_size: int = FLA_CHUNK_SIZE,
):
    """u = (I+A)^-1 (beta * v); w = (I+A)^-1 (beta * exp(g) * k)."""
    if not k.is_contiguous():
        k = k.contiguous()
    if not v.is_contiguous():
        v = v.contiguous()
    B, T, Hg, K = k.shape
    H, V = v.shape[-2], v.shape[-1]
    BT = chunk_size
    # w carries one head per *value* head (not per key head): the kernel writes it
    # with the (H * K) token stride of the value heads.
    w = torch.empty(B, T, H, K, dtype=k.dtype, device=k.device)
    u = torch.empty_like(v)
    NT = chunk_indices.shape[0] if cu_seqlens is not None else triton.cdiv(T, BT)
    BK = min(triton.next_power_of_2(K), 64)
    BV = min(triton.next_power_of_2(V), 64)
    _recompute_w_u_fwd_kernel[(NT, B * H)](
        k, v, beta, w, u, A, g_cumsum, cu_seqlens, chunk_indices,
        T=T, H=H, Hg=Hg, K=K, V=V, BT=BT, BK=BK, BV=BV,
        IS_VARLEN=cu_seqlens is not None, num_warps=4,
    )
    return w, u


@triton.jit(do_not_specialize=["T"])
def _chunk_gated_delta_rule_fwd_h_kernel(
    k,
    v,
    w,
    v_new,
    g,
    h,
    h0,
    ht,
    cu_seqlens,
    chunk_offsets,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """Sequential scan over chunks; h[i_t] holds the state *before* chunk i_t."""
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    # [BV, BK] state, K split into 64-wide slices (K <= 256)
    b_h1 = tl.zeros([BV, 64], dtype=tl.float32)
    if K > 64:
        b_h2 = tl.zeros([BV, 64], dtype=tl.float32)
    if K > 128:
        b_h3 = tl.zeros([BV, 64], dtype=tl.float32)
    if K > 192:
        b_h4 = tl.zeros([BV, 64], dtype=tl.float32)

    h += ((boh * H + i_h) * V * K).to(tl.int64)
    v += ((bos * H + i_h) * V).to(tl.int64)
    k += ((bos * Hg + i_h // (H // Hg)) * K).to(tl.int64)
    w += ((bos * H + i_h) * K).to(tl.int64)
    v_new += ((bos * H + i_h) * V).to(tl.int64)
    stride_v = H * V
    stride_h = H * V * K
    stride_k = Hg * K
    stride_w = H * K
    if USE_INITIAL_STATE:
        h0 = h0 + i_nh * V * K
    if STORE_FINAL_STATE:
        ht = ht + i_nh * V * K

    if USE_INITIAL_STATE:
        p_h0_1 = tl.make_block_ptr(h0, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
        b_h1 += tl.load(p_h0_1, boundary_check=(0, 1)).to(tl.float32)
        if K > 64:
            p_h0_2 = tl.make_block_ptr(h0, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0))
            b_h2 += tl.load(p_h0_2, boundary_check=(0, 1)).to(tl.float32)
        if K > 128:
            p_h0_3 = tl.make_block_ptr(h0, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0))
            b_h3 += tl.load(p_h0_3, boundary_check=(0, 1)).to(tl.float32)
        if K > 192:
            p_h0_4 = tl.make_block_ptr(h0, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0))
            b_h4 += tl.load(p_h0_4, boundary_check=(0, 1)).to(tl.float32)

    for i_t in range(NT):
        p_h1 = tl.make_block_ptr(
            h + i_t.to(tl.int64) * stride_h, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0)
        )
        tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_h2 = tl.make_block_ptr(
                h + i_t.to(tl.int64) * stride_h, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0)
            )
            tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            p_h3 = tl.make_block_ptr(
                h + i_t.to(tl.int64) * stride_h, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0)
            )
            tl.store(p_h3, b_h3.to(p_h3.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            p_h4 = tl.make_block_ptr(
                h + i_t.to(tl.int64) * stride_h, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0)
            )
            tl.store(p_h4, b_h4.to(p_h4.dtype.element_ty), boundary_check=(0, 1))

        p_w = tl.make_block_ptr(w, (T, K), (stride_w, 1), (i_t * BT, 0), (BT, 64), (1, 0))
        b_w = tl.load(p_w, boundary_check=(0, 1))
        b_v = tl.dot(b_w, tl.trans(b_h1).to(b_w.dtype))
        if K > 64:
            p_w = tl.make_block_ptr(w, (T, K), (stride_w, 1), (i_t * BT, 64), (BT, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, tl.trans(b_h2).to(b_w.dtype))
        if K > 128:
            p_w = tl.make_block_ptr(w, (T, K), (stride_w, 1), (i_t * BT, 128), (BT, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, tl.trans(b_h3).to(b_w.dtype))
        if K > 192:
            p_w = tl.make_block_ptr(w, (T, K), (stride_w, 1), (i_t * BT, 192), (BT, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, tl.trans(b_h4).to(b_w.dtype))
        p_v = tl.make_block_ptr(v, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1)) - b_v

        p_vn = tl.make_block_ptr(
            v_new, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        tl.store(p_vn, b_v.to(p_vn.dtype.element_ty), boundary_check=(0, 1))

        last_idx = min((i_t.to(tl.int64) + 1) * BT, T) - 1
        m_t = (i_t.to(tl.int64) * BT + tl.arange(0, BT)) < T
        b_g_last = tl.load(g + bos * H + last_idx * H + i_h)
        p_g = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
        b_g = tl.load(p_g, boundary_check=(0,))
        b_v = b_v * tl.where(m_t, tl.exp(b_g_last - b_g), 0)[:, None]
        b_g_last = tl.exp(b_g_last)
        b_h1 *= b_g_last
        if K > 64:
            b_h2 *= b_g_last
        if K > 128:
            b_h3 *= b_g_last
        if K > 192:
            b_h4 *= b_g_last

        b_v = b_v.to(k.dtype.element_ty)
        p_k = tl.make_block_ptr(k, (K, T), (1, stride_k), (0, i_t * BT), (64, BT), (0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_h1 += tl.trans(tl.dot(b_k, b_v))
        if K > 64:
            p_k = tl.make_block_ptr(k, (K, T), (1, stride_k), (64, i_t * BT), (64, BT), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h2 += tl.trans(tl.dot(b_k, b_v))
        if K > 128:
            p_k = tl.make_block_ptr(k, (K, T), (1, stride_k), (128, i_t * BT), (64, BT), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h3 += tl.trans(tl.dot(b_k, b_v))
        if K > 192:
            p_k = tl.make_block_ptr(k, (K, T), (1, stride_k), (192, i_t * BT), (64, BT), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h4 += tl.trans(tl.dot(b_k, b_v))

    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(ht, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
        tl.store(p_ht, b_h1.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_ht = tl.make_block_ptr(ht, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0))
            tl.store(p_ht, b_h2.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            p_ht = tl.make_block_ptr(ht, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0))
            tl.store(p_ht, b_h3.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            p_ht = tl.make_block_ptr(ht, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0))
            tl.store(p_ht, b_h4.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


def chunk_gated_delta_rule_fwd_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
    chunk_offsets: torch.Tensor,
    chunk_size: int = FLA_CHUNK_SIZE,
):
    B, T, Hg, K = k.shape
    H, V = u.shape[-2], u.shape[-1]
    BT = chunk_size
    N, NT = len(cu_seqlens) - 1, len(chunk_indices)
    assert K <= 256, "chunk_delta_h supports head_k_dim <= 256"

    h = torch.empty(B, NT, H, V, K, dtype=k.dtype, device=k.device)
    v_new = torch.empty_like(u)
    final_state = (
        torch.empty(N, H, V, K, dtype=torch.float32, device=k.device)
        if output_final_state
        else None
    )
    BV = min(triton.next_power_of_2(V), 64)
    grid = (triton.cdiv(V, BV), N * H)
    _chunk_gated_delta_rule_fwd_h_kernel[grid](
        k=k,
        v=u,
        w=w,
        v_new=v_new,
        g=g,
        h=h,
        h0=initial_state,
        ht=final_state if output_final_state else None,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        BV=BV,
        USE_INITIAL_STATE=initial_state is not None,
        STORE_FINAL_STATE=output_final_state,
        IS_VARLEN=True,
        num_warps=4,
        num_stages=3,
    )
    return h, v_new, final_state


@triton.jit(do_not_specialize=["T"])
def _chunk_fwd_o_kernel(
    q,
    k,
    v,
    h,
    g,
    o,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_tg = i_t
        i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
        i_t = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    q += (bos * Hg + i_h // (H // Hg)) * K
    k += (bos * Hg + i_h // (H // Hg)) * K
    v += (bos * H + i_h) * V
    o += (bos * H + i_h) * V
    h += (i_tg * H + i_h).to(tl.int64) * V * K

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_A = tl.zeros([BT, BT], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(q, (T, K), (Hg * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k, (K, T), (1, Hg * K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_h = tl.make_block_ptr(h, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_o += tl.dot(b_q, tl.trans(b_h))
        b_A += tl.dot(b_q, b_k)

    g += bos * H + i_h
    p_g = tl.make_block_ptr(g, (T,), (H,), (i_t * BT,), (BT,), (0,))
    b_g = tl.load(p_g, boundary_check=(0,))
    b_o = b_o * tl.exp(b_g)[:, None]
    b_A = b_A * tl.exp(b_g[:, None] - b_g[None, :])

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
    b_A = tl.where(m_A, b_A, 0)

    p_v = tl.make_block_ptr(v, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_o = tl.make_block_ptr(o, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_o = b_o * scale + tl.dot(b_A.to(b_v.dtype), b_v) * scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


def chunk_fwd_o(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    g: torch.Tensor,
    scale: float,
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
    chunk_size: int = FLA_CHUNK_SIZE,
) -> torch.Tensor:
    if not q.is_contiguous():
        q = q.contiguous()
    if not k.is_contiguous():
        k = k.contiguous()
    if not v.is_contiguous():
        v = v.contiguous()
    B, T, Hg, K = q.shape
    H, V = v.shape[-2], v.shape[-1]
    BT = chunk_size
    NT = len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T, BT)
    o = torch.empty_like(v)
    BK = min(triton.next_power_of_2(K), 64)
    BV = min(triton.next_power_of_2(V), 64)
    _chunk_fwd_o_kernel[(triton.cdiv(V, BV), NT, B * H)](
        q, k, v, h, g, o, cu_seqlens, chunk_indices, scale,
        T=T, H=H, Hg=Hg, K=K, V=V, BT=BT, BK=BK, BV=BV,
        IS_VARLEN=cu_seqlens is not None, num_warps=4, num_stages=3,
    )
    return o


def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor | None = None,
    chunk_offsets: torch.Tensor | None = None,
    scale: float | None = None,
    chunk_size: int = FLA_CHUNK_SIZE,
    return_h: bool = False,
):
    """Chunked gated delta rule over a varlen batch.

    Args:
        q/k:   [1, T, Hg, K]
        v:     [1, T, H, V]
        g:     [1, T, H]  log-decay, <= 0
        beta:  [1, T, H]
        initial_state: [N, H, V, K] float32 or None
        cu_seqlens: [N+1] int32
        return_h: also hand back the per-chunk prefix states the scan already
            computes (`h[global_chunk]` is the state *before* that chunk), which
            the prefix cache snapshots mid-sequence.
    Returns:
        o: [1, T, H, V], final_state: [N, H, V, K] float32 or None,
        and h: [1, NT, H, V, K] when ``return_h``
    """
    assert q.dtype == k.dtype == v.dtype, "q/k/v must share a dtype"
    assert q.dtype != torch.float32, "chunked GDN only supports bf16/fp16 inputs"
    assert beta.shape == g.shape
    if cu_seqlens is None:
        raise ValueError("this port requires cu_seqlens (varlen) input")
    if q.shape[0] != 1:
        raise ValueError("with cu_seqlens the batch dim must be 1 (sequences packed on T)")
    if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
        raise ValueError("initial_state must hold one state per sequence")
    # the kernels read the head counts off different tensors (H from v/beta/g, Hg from
    # q/k) and index key head i_h // (H // Hg): swapped or non-multiple counts would walk
    # off g/beta silently instead of failing
    assert q.shape[2] == k.shape[2], "q and k must share a key head count"
    assert beta.shape[-1] == g.shape[-1] == v.shape[-2], \
        "beta/g must be per value head ([*, H]) and match v"
    assert v.shape[-2] % k.shape[2] == 0 and v.shape[-2] >= k.shape[2], \
        "value heads must be a positive multiple of key heads (GQA)"
    assert initial_state is None or (initial_state.dtype == torch.float32
                                     and initial_state.is_contiguous()), \
        "initial_state is read with hardcoded strides: must be contiguous fp32"

    if chunk_indices is None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    if chunk_offsets is None:
        chunk_offsets = prepare_chunk_offsets(cu_seqlens, chunk_size)
    if scale is None:
        scale = k.shape[-1] ** -0.5

    g_cum = chunk_local_cumsum(g, cu_seqlens, chunk_indices, chunk_size)
    A = chunk_scaled_dot_kkt_fwd(
        k=k, beta=beta, g=g_cum, cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices, chunk_size=chunk_size, output_dtype=torch.float32,
    )
    Ai = solve_tril(A, cu_seqlens, chunk_indices, output_dtype=k.dtype)
    w, u = recompute_w_u_fwd(
        k=k, v=v, beta=beta, g_cumsum=g_cum, A=Ai,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, chunk_size=chunk_size,
    )
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=k, w=w, u=u, g=g_cum, initial_state=initial_state,
        output_final_state=output_final_state, cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices, chunk_offsets=chunk_offsets, chunk_size=chunk_size,
    )
    o = chunk_fwd_o(
        q=q, k=k, v=v_new, h=h, g=g_cum, scale=scale,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, chunk_size=chunk_size,
    )
    if return_h:
        return o, final_state, h
    return o, final_state


# ---------------------------------------------------------------------------
# decode -- one token per sequence, state updated in place
#
# decode -- one kernel fusing the gating (g = -exp(A_log) * softplus(a + dt_bias),
# beta = sigmoid(b)), the q/k l2norm and the delta-rule step
# ---------------------------------------------------------------------------
@triton.jit
def _fused_recurrent_gdn_decode_kernel(
    mixed_qkv,
    a,
    b,
    A_log,
    dt_bias,
    o,
    h0,                      # [num_slots, HV, V, K] fp32, read *and* written
    ssm_state_indices,
    scale,
    stride_mixed_qkv_tok: tl.constexpr,
    stride_a_tok: tl.constexpr,
    stride_b_tok: tl.constexpr,
    stride_state_token: tl.constexpr,
    stride_indices_seq: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    SOFTPLUS_THRESHOLD: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]

    state_idx = tl.load(ssm_state_indices + i_n * stride_indices_seq).to(tl.int64)
    p_o = o + (i_n * HV + i_hv) * V + o_v
    if state_idx < 0:
        # padded row: ModelRunner fills the CUDA-graph batch padding with -1.
        # Emit zeros and leave the pool untouched (slot 0 is a normal slot here).
        tl.store(p_o, tl.zeros([BV], dtype=o.dtype.element_ty), mask=mask_v)
        return

    p_h0 = h0 + state_idx * stride_state_token
    p_h0 = p_h0 + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
    b_h = tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    p_mixed = mixed_qkv + i_n * stride_mixed_qkv_tok
    q_off = i_h * K + o_k
    k_off = (H * K) + i_h * K + o_k
    v_off = (2 * H * K) + i_hv * V + o_v
    b_q = tl.load(p_mixed + q_off, mask=mask_k, other=0).to(tl.float32)
    b_k = tl.load(p_mixed + k_off, mask=mask_k, other=0).to(tl.float32)
    b_v = tl.load(p_mixed + v_off, mask=mask_v, other=0).to(tl.float32)

    if USE_QK_L2NORM_IN_KERNEL:
        b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
        b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
    b_q = b_q * scale

    a_val = tl.load(a + i_n * stride_a_tok + i_hv).to(tl.float32)
    b_val = tl.load(b + i_n * stride_b_tok + i_hv).to(tl.float32)
    A_log_val = tl.load(A_log + i_hv).to(tl.float32)
    dt_bias_val = tl.load(dt_bias + i_hv).to(tl.float32)
    x = a_val + dt_bias_val
    softplus_x = tl.where(x <= SOFTPLUS_THRESHOLD, tl.log(1.0 + tl.exp(x)), x)
    g_val = -tl.exp(A_log_val) * softplus_x
    beta_val = tl.sigmoid(b_val).to(b.dtype.element_ty).to(tl.float32)

    b_h *= tl.exp(g_val)
    b_v -= tl.sum(b_h * b_k[None, :], 1)
    b_v *= beta_val
    b_h += b_v[:, None] * b_k[None, :]
    b_o = tl.sum(b_h * b_q[None, :], 1)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)
    tl.store(p_h0, b_h.to(p_h0.dtype.element_ty), mask=mask_h)


def fused_recurrent_gdn_decode(
    mixed_qkv: torch.Tensor,      # [B, 2*H*K + HV*V], post-conv
    a: torch.Tensor,              # [B, HV]
    b: torch.Tensor,              # [B, HV]
    A_log: torch.Tensor,          # [HV]
    dt_bias: torch.Tensor,        # [HV]
    ssm_state: torch.Tensor,      # [num_slots, HV, V, K] fp32, updated in place
    state_idx: torch.Tensor,      # [B] int32
    scale: float | None = None,
    use_qk_l2norm_in_kernel: bool = True,
) -> torch.Tensor:
    B = mixed_qkv.shape[0]
    assert mixed_qkv.stride(-1) == 1, "mixed_qkv must be contiguous in the last dim"
    HV, V, K = ssm_state.shape[-3:]
    qk_dim = mixed_qkv.shape[1] - HV * V
    assert qk_dim > 0 and qk_dim % 2 == 0, "mixed_qkv last dim mismatch"
    q_dim = qk_dim // 2
    assert q_dim % K == 0, "mixed_qkv Q size is not a multiple of head_k_dim"
    H = q_dim // K
    assert HV % H == 0, "num_value_heads must be a multiple of num_key_heads"
    if scale is None:
        scale = K ** -0.5
    BK = triton.next_power_of_2(K)
    assert triton.cdiv(K, BK) == 1, "decode kernel expects head_k_dim to fit one block"
    BV = min(triton.next_power_of_2(V), 32)
    out = torch.empty(B, HV, V, dtype=mixed_qkv.dtype, device=mixed_qkv.device)
    grid = (triton.cdiv(V, BV), B * HV)
    _fused_recurrent_gdn_decode_kernel[grid](
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        o=out,
        h0=ssm_state,
        ssm_state_indices=state_idx,
        scale=scale,
        stride_mixed_qkv_tok=mixed_qkv.stride(0),
        stride_a_tok=a.stride(0),
        stride_b_tok=b.stride(0),
        stride_state_token=ssm_state.stride(0),
        stride_indices_seq=state_idx.stride(0),
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        SOFTPLUS_THRESHOLD=SOFTPLUS_THRESHOLD,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        num_warps=1,
        num_stages=3,
    )
    return out
