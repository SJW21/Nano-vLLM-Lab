"""Grouped-GEMM MoE (`moe_tp`): one compute path for eager and CUDA graph.

Weights are K-major and expert-local: `w1[e]` is `(H, 2I_local)` and `w2[e]` is
`(I_local, H)`, where `I_local = moe_intermediate_size / ep_size`. The kernel
walks the experts in one persistent launch (virtual-SM scheduling: grid is
`NUM_SM`, a program strides over tiles with `tile_idx += NUM_SM`), so the launch
topology does not depend on the batch or on the routing.

Routing is sorted on the device (`argsort(stable=True)`), counted with a
`scatter_add_` histogram, and turned into an exclusive prefix sum with
`cumsum`. The kernels read the per-expert work size from the device `counts`
tensor and locate each expert's rows with `row_offsets`; no host-side read of
the routing is ever needed. That is what makes the whole path capturable: a
host read (`.item()`/`.tolist()`) inside a capture aborts it, and a host-built
pointer table baked into the graph silently reads the capture-time values on
replay.

Allocation: every call sizes its workspace from the **actual number of routed
token-expert pairs**, `T * top_k`. Per-expert `M_i` are device values and only
drive the number of tiles, never an allocation size -- using them for shapes
would require a device-to-host sync. The workspace is created with plain
`torch.empty` inside `moe_forward`; there is no cross-layer buffer pool.

CUDA graph: the same Python function runs inside and outside capture. When it
runs inside `torch.cuda.graph(pool=...)`, the allocations land in that graph's
private memory pool, so replay reuses the same addresses. Outside capture they
land in the normal caching allocator. The only rule the caller must respect is
that the returned tensor is consumed before the next replay (it is a view of
the caller's graph-pool memory when captured).
"""

import torch

import triton
import triton.language as tl

__all__ = [
    "MoEWeights", "moe_forward",
    "grouped_matmul_kernel_masked", "grouped_matmul_silu_kernel_masked",
    "DEVICE",
]

DEVICE = torch.device("cuda")


def num_sms() -> int:
    """SM count of the active device.

    This runs at import time (the autotune lists below are module-level constants),
    so it falls back rather than raising when there is no live CUDA device. That
    only affects the autotune candidate set: NUM_SM is a grid size, so any value
    is functionally correct, just potentially slower.
    """
    try:
        return torch.cuda.get_device_properties("cuda").multi_processor_count
    except Exception:
        return 132


# =============================================================================
# 1. Autotune configuration
# =============================================================================
#
# Two lists, one per kernel. They must be kept **symmetric**: the only reason to
# have two is that the fused kernel carries two fp32 accumulators instead of one
# and therefore has different register pressure. If the fused list is narrower
# in some tile, that shows up as a bogus "fusion is slower" result.

_GEMM_CONFIGS = [
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'NUM_SM': 84}),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'NUM_SM': 128}),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'NUM_SM': 84}),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'NUM_SM': 128}),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'NUM_SM': num_sms()}),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'NUM_SM': num_sms()}),
    # The three entries below carry num_warps=8. The six above do not, so they
    # all fall back to Triton's default of 4 warps -- under which a 128x128 fp32
    # accumulator (16384 registers, 128/thread) plus operands necessarily spills,
    # and autotune simply never picks the large tiles. Adding these made a
    # 128/128/64 tile reachable and took the masked kernel from 8.17ms to 6.57ms.
    #
    # Keep them in sync with _SILU_GEMM_CONFIGS: with a fused kernel carrying two
    # accumulators (32768 registers at 128x128) the 8-warp variants are what make
    # that tile possible at all. Narrowing only one list would rig the comparison.
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32,
                   'NUM_SM': num_sms()}, num_warps=8),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64,
                   'NUM_SM': num_sms()}, num_warps=8),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64,
                   'NUM_SM': num_sms()}, num_warps=8),
]

# Fused gate_up + SiLU. The K/N block sizes here must not exceed MAX_BLOCK_K /
# MAX_BLOCK_N below, or the even-flag predicates would judge the *unfused*
# limits against a fused config and drop a mask that is still needed.
_SILU_GEMM_CONFIGS = [
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'NUM_SM': 84}),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'NUM_SM': 128}),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'NUM_SM': num_sms()}),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'NUM_SM': num_sms()}),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'NUM_SM': num_sms()}),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'NUM_SM': num_sms()},
                  num_warps=8),
    # Mirrors the last two entries of _GEMM_CONFIGS (see the note there).
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32,
                   'NUM_SM': num_sms()}, num_warps=8),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64,
                   'NUM_SM': num_sms()}, num_warps=8),
]


def _max_block(configs, name: str) -> int:
    return max(c.kwargs[name] for c in configs)


# Element types the Triton kernels can be specialised on. Both are 2 bytes, so
# the `stride(0) % 16 == 0` alignment hints below (which assume 16 *elements* per
# vector) are equivalent for the two.
_TRITON_DTYPE = {
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
}


def _triton_dtype(torch_dtype: torch.dtype):
    """Map a torch dtype to the Triton constexpr the kernels are specialised on."""
    try:
        return _TRITON_DTYPE[torch_dtype]
    except KeyError:
        raise TypeError(
            f"unsupported dtype {torch_dtype}; the Triton kernels are specialised "
            f"for {sorted(str(d) for d in _TRITON_DTYPE)}"
        ) from None


# Largest BLOCK_SIZE per axis across both config lists. A dimension that is a
# multiple of these is a multiple of every config's block size, which is exactly
# what EVEN_K / EVEN_N test for -- so this has to be the **max**, over the union
# of both lists. A smaller bound would let a config with a larger block size pass
# the predicate with a mask it still needs, i.e. an out-of-bounds store.
MAX_BLOCK_N = max(_max_block(_GEMM_CONFIGS, 'BLOCK_SIZE_N'),
                  _max_block(_SILU_GEMM_CONFIGS, 'BLOCK_SIZE_N'))
MAX_BLOCK_K = max(_max_block(_GEMM_CONFIGS, 'BLOCK_SIZE_K'),
                  _max_block(_SILU_GEMM_CONFIGS, 'BLOCK_SIZE_K'))


# =============================================================================
# 2. Kernels
# =============================================================================

@triton.autotune(configs=_GEMM_CONFIGS, key=['group_size'])
@triton.jit
def grouped_matmul_kernel_masked(
    A_ptr, b_ptrs, C_ptr,
    # [E] device tensors: rows routed to each expert, and their exclusive prefix
    # sum (the first row of expert g). Both are produced on the device each call.
    counts, row_offsets,
    # Element strides. All experts share one shape, so these are scalar.
    lda, ldb, ldc,
    group_size,
    K: tl.constexpr,
    N: tl.constexpr,
    NUM_SM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    EVEN_N: tl.constexpr,
    # Element type of A, B and C. Specialising on it (rather than hard-coding
    # fp16) is what lets the same kernel serve fp16 and bf16 -- both are 2-byte
    # MMA input types and both accumulate in fp32.
    DTYPE: tl.constexpr,
):
    tile_idx = tl.program_id(0)
    last_problem_end = 0
    for g in range(group_size):
        gm = tl.load(counts + g)
        num_m_tiles = tl.cdiv(gm, BLOCK_SIZE_M)
        num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
        num_tiles = num_m_tiles * num_n_tiles
        while (tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles):
            # row0 must be int64: rows * stride can exceed int32 for prefill.
            row0 = tl.load(row_offsets + g).to(tl.int64)
            a_ptr = A_ptr + row0 * lda
            b_ptr = tl.load(b_ptrs + g).to(tl.pointer_type(DTYPE))
            c_ptr = C_ptr + row0 * ldc

            tile_idx_in_gemm = tile_idx - last_problem_end
            tile_m_idx = tile_idx_in_gemm // num_n_tiles
            tile_n_idx = tile_idx_in_gemm % num_n_tiles

            offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            offs_k = tl.arange(0, BLOCK_SIZE_K)
            # The M mask is the only one that must survive: gm is a runtime value.
            # It is computed unconditionally but, being uniform along k, does not
            # break vectorization. mask_bn is dead code unless EVEN_N is false.
            mask_am = offs_am < gm
            mask_bn = offs_bn < N
            a_ptrs = a_ptr + offs_am[:, None] * lda + offs_k[None, :]
            b_tile_ptrs = b_ptr + offs_k[:, None] * ldb + offs_bn[None, :]
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            for kk in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
                if EVEN_K:
                    # K aligned -> drop the K mask entirely, and restore the
                    # alignment hint that lets Triton vectorize the load.
                    tl.multiple_of(a_ptrs, [16, 16])
                    a = tl.load(a_ptrs, mask=mask_am[:, None], other=0.0)
                    if EVEN_N:
                        tl.multiple_of(b_tile_ptrs, [16, 16])
                        b = tl.load(b_tile_ptrs)
                    else:
                        b = tl.load(b_tile_ptrs, mask=mask_bn[None, :], other=0.0)
                else:
                    mask_k = offs_k < K - kk * BLOCK_SIZE_K
                    a = tl.load(a_ptrs, mask=mask_am[:, None] & mask_k[None, :], other=0.0)
                    if EVEN_N:
                        b = tl.load(b_tile_ptrs, mask=mask_k[:, None], other=0.0)
                    else:
                        b = tl.load(b_tile_ptrs,
                                    mask=mask_k[:, None] & mask_bn[None, :], other=0.0)
                accumulator += tl.dot(a, b)
                a_ptrs += BLOCK_SIZE_K
                b_tile_ptrs += BLOCK_SIZE_K * ldb
            c = accumulator.to(DTYPE)

            offs_cm = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_cn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            c_ptrs = c_ptr + ldc * offs_cm[:, None] + offs_cn[None, :]
            # Out-of-range positions must not be stored: that corrupts whatever
            # lives next to C in memory (and in the N direction it wraps around
            # and overwrites the tile's own next row).
            if EVEN_N:
                tl.store(c_ptrs, c, mask=(offs_cm < gm)[:, None])
            else:
                tl.store(c_ptrs, c, mask=(offs_cm < gm)[:, None] & (offs_cn < N)[None, :])

            tile_idx += NUM_SM

        last_problem_end = last_problem_end + num_tiles


# -----------------------------------------------------------------------------
# Fused epilogue: gate_up GEMM + SiLU, in one kernel
# -----------------------------------------------------------------------------
#
# B is the merged (gate | up) weight, row-major with B's N = 2I contiguous.
# `N` below is B's N; the output width is `half = N // 2`. A tile of A is shared
# by the gate and up dots, which is where the fusion saves its traffic.

@triton.autotune(configs=_SILU_GEMM_CONFIGS, key=['group_size'])
@triton.jit
def grouped_matmul_silu_kernel_masked(
    A_ptr, b_ptrs, C_ptr,
    counts, row_offsets,
    lda, ldb, ldc,
    group_size,
    K: tl.constexpr,
    N: tl.constexpr,          # B's N = 2 * I_local; output width is N // 2
    NUM_SM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    # EVEN_N here tests the **output** width half = N // 2, because the tiles are
    # laid out over the output, not over B's N.
    EVEN_K: tl.constexpr,
    EVEN_N: tl.constexpr,
    DTYPE: tl.constexpr,
):
    tile_idx = tl.program_id(0)
    last_problem_end = 0
    half = N // 2
    for g in range(group_size):
        gm = tl.load(counts + g)
        num_m_tiles = tl.cdiv(gm, BLOCK_SIZE_M)
        num_n_tiles = tl.cdiv(half, BLOCK_SIZE_N)
        num_tiles = num_m_tiles * num_n_tiles
        while (tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles):
            row0 = tl.load(row_offsets + g).to(tl.int64)
            a_ptr = A_ptr + row0 * lda
            b_ptr = tl.load(b_ptrs + g).to(tl.pointer_type(DTYPE))
            c_ptr = C_ptr + row0 * ldc

            tile_idx_in_gemm = tile_idx - last_problem_end
            tile_m_idx = tile_idx_in_gemm // num_n_tiles
            tile_n_idx = tile_idx_in_gemm % num_n_tiles

            offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            # offs_bn is an **output** column index, in [0, half)
            offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            offs_k = tl.arange(0, BLOCK_SIZE_K)
            mask_am = offs_am < gm
            mask_bn = offs_bn < half

            a_ptrs = a_ptr + offs_am[:, None] * lda + offs_k[None, :]
            # B is (K, N) row-major with N contiguous, so the two slices start at
            # offs_bn and offs_bn + half along the contiguous axis. offs_bn < half
            # guarantees the up columns stay below N and never run out of range.
            bg_ptrs = b_ptr + offs_k[:, None] * ldb + offs_bn[None, :]
            bu_ptrs = b_ptr + offs_k[:, None] * ldb + (offs_bn + half)[None, :]

            acc_g = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            acc_u = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            for kk in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
                if EVEN_K:
                    tl.multiple_of(a_ptrs, [16, 16])
                    a = tl.load(a_ptrs, mask=mask_am[:, None], other=0.0)
                    if EVEN_N:
                        # N aligned -> neither B slice needs a mask, and
                        # half % BLOCK_SIZE_N == 0 keeps offs_bn + half aligned
                        tl.multiple_of(bg_ptrs, [16, 16])
                        tl.multiple_of(bu_ptrs, [16, 16])
                        bg = tl.load(bg_ptrs)
                        bu = tl.load(bu_ptrs)
                    else:
                        bg = tl.load(bg_ptrs, mask=mask_bn[None, :], other=0.0)
                        bu = tl.load(bu_ptrs, mask=mask_bn[None, :], other=0.0)
                else:
                    mask_k = offs_k < K - kk * BLOCK_SIZE_K
                    a = tl.load(a_ptrs, mask=mask_am[:, None] & mask_k[None, :], other=0.0)
                    if EVEN_N:
                        bg = tl.load(bg_ptrs, mask=mask_k[:, None], other=0.0)
                        bu = tl.load(bu_ptrs, mask=mask_k[:, None], other=0.0)
                    else:
                        bg = tl.load(bg_ptrs, mask=mask_k[:, None] & mask_bn[None, :], other=0.0)
                        bu = tl.load(bu_ptrs, mask=mask_k[:, None] & mask_bn[None, :], other=0.0)
                # The A tile is shared by both dots -- this is what lets the fused
                # kernel halve A's read traffic.
                acc_g += tl.dot(a, bg)
                acc_u += tl.dot(a, bu)
                a_ptrs += BLOCK_SIZE_K
                bg_ptrs += BLOCK_SIZE_K * ldb
                bu_ptrs += BLOCK_SIZE_K * ldb

            # silu(x) = x * sigmoid(x), written multiplicatively to save a temp.
            # Computed in fp32, which is more accurate than the unfused path.
            c = (acc_g * tl.sigmoid(acc_g)) * acc_u
            c = c.to(DTYPE)

            offs_cm = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            c_ptrs = c_ptr + ldc * offs_cm[:, None] + offs_bn[None, :]
            if EVEN_N:
                tl.store(c_ptrs, c, mask=(offs_cm < gm)[:, None])
            else:
                tl.store(c_ptrs, c, mask=(offs_cm < gm)[:, None] & (offs_bn < half)[None, :])

            tile_idx += NUM_SM

        last_problem_end = last_problem_end + num_tiles


# =============================================================================
# 3. Weights
# =============================================================================

class MoEWeights:
    """Expert weights in K-major layout, prepared once at model construction.

    Args:
        w1: (E, 2I, H) -- merged gate/up projection, nn.Linear layout.
        w2: (E, H, I)  -- down projection.

    Each expert is stored as its own contiguous (K, N) tensor. That is why the
    kernel needs a device pointer table (`w1_b_ptrs` / `w2_b_ptrs`): one
    `data_ptr()` per expert, built once here and never rebuilt. A later
    refactoring can stack the experts into one `(E, K, N)` tensor and replace
    the table with `base + g * K * N`; until then the table is the only
    address-dependent state.
    """

    def __init__(self, w1: torch.Tensor, w2: torch.Tensor):
        assert w1.dim() == 3 and w2.dim() == 3, "expected (E, out, in) tensors"
        assert w1.dtype == w2.dtype, (
            f"w1 and w2 must share one dtype, got {w1.dtype} / {w2.dtype}")
        self.dtype = w1.dtype
        _triton_dtype(self.dtype)         # reject anything the kernels cannot take
        num_experts, two_i, hidden = w1.shape
        assert w2.shape == (num_experts, hidden, two_i // 2), (
            f"w2 shape {tuple(w2.shape)} is not the (E, H, I) matching "
            f"w1 {tuple(w1.shape)}")
        self.num_experts = num_experts
        self.hidden_size = hidden
        self.intermediate_size = two_i // 2
        # non-TMA kernels want B as (K, N) contiguous; the weights are fixed, so
        # this transpose happens here and never inside the forward pass
        self.w1 = [w1[e].t().contiguous() for e in range(num_experts)]   # (H, 2I)
        self.w2 = [w2[e].t().contiguous() for e in range(num_experts)]   # (I, H)
        device = w1.device
        self.w1_b_ptrs = torch.tensor(
            [w.data_ptr() for w in self.w1], dtype=torch.int64, device=device)
        self.w2_b_ptrs = torch.tensor(
            [w.data_ptr() for w in self.w2], dtype=torch.int64, device=device)

    def __repr__(self):
        mb = sum(w.numel() * w.element_size() for w in self.w1 + self.w2) / 1e6
        return (f"MoEWeights(E={self.num_experts}, H={self.hidden_size}, "
                f"I={self.intermediate_size}, {mb:.0f}MB {self.dtype})")


# =============================================================================
# 4. Workspace and forward
# =============================================================================

class _MoEWorkspace:
    """All intermediates of one MoE forward, sized by `T * top_k`.

    The object lives only for the duration of the call. In eager mode its
    tensors are returned to the caching allocator when it goes out of scope; in
    a CUDA graph capture they are owned by the graph's private pool.
    """

    __slots__ = ("permuted", "act", "down", "final", "counts", "row_offsets",
                 "token_idx", "sorted_weights", "count_f", "ones_f",
                 "scratch", "acc", "rows")


def _alloc_workspace(num_tokens, top_k, hidden_size, intermediate_size,
                     num_experts, dtype, device) -> _MoEWorkspace:
    rows = num_tokens * top_k
    ws = _MoEWorkspace()
    ws.rows = rows
    ws.permuted = torch.empty((rows, hidden_size), dtype=dtype, device=device)
    ws.act = torch.empty((rows, intermediate_size), dtype=dtype, device=device)
    ws.down = torch.empty((rows, hidden_size), dtype=dtype, device=device)
    ws.final = torch.empty((num_tokens, hidden_size), dtype=dtype, device=device)
    ws.counts = torch.empty((num_experts,), dtype=torch.int32, device=device)
    ws.row_offsets = torch.empty((num_experts,), dtype=torch.int32, device=device)
    ws.token_idx = torch.empty((rows,), dtype=torch.int64, device=device)
    ws.sorted_weights = torch.empty((rows,), dtype=dtype, device=device)
    # fp32 histogram: torch.bincount computes its output length from max(input),
    # which calls .item() and aborts a capture. scatter_add_ is the capturable
    # equivalent; counts are integers and fp32 represents them exactly below
    # 2^24, which is far above any batch we schedule.
    ws.count_f = torch.zeros((num_experts,), dtype=torch.float32, device=device)
    ws.ones_f = torch.ones((rows,), dtype=torch.float32, device=device)
    ws.scratch = torch.empty((rows, hidden_size), dtype=torch.float32, device=device)
    ws.acc = torch.empty((num_tokens, hidden_size), dtype=torch.float32, device=device)
    return ws


def _align_into(ws: _MoEWorkspace, hidden_states, topk_weights, topk_ids, top_k):
    """Stage 1: permutation + histogram + exclusive prefix sum. Device-side."""
    n = hidden_states.shape[0] * top_k
    flat = topk_ids.reshape(-1)
    order = torch.argsort(flat, stable=True)
    ws.token_idx[:n].copy_(torch.div(order, top_k, rounding_mode="floor"))
    ws.sorted_weights[:n].copy_(topk_weights.reshape(-1).index_select(0, order))
    torch.index_select(hidden_states, 0, ws.token_idx[:n], out=ws.permuted[:n])

    ws.count_f.zero_()
    ws.count_f.scatter_add_(0, flat, ws.ones_f[:n])
    ws.counts.copy_(ws.count_f)                  # fp32 -> int32, exact
    ws.row_offsets.copy_(torch.cumsum(ws.counts, 0) - ws.counts)


def _unpermute_into(ws: _MoEWorkspace, num_tokens, top_k):
    """Stage 4: weighted scatter-add back to (T, H)."""
    n = num_tokens * top_k
    w = ws.sorted_weights[:n].to(torch.float32).unsqueeze(-1)
    torch.mul(ws.down[:n], w, out=ws.scratch[:n])
    ws.acc[:num_tokens].zero_()
    ws.acc[:num_tokens].index_add_(0, ws.token_idx[:n], ws.scratch[:n])
    return ws.final[:num_tokens].copy_(ws.acc[:num_tokens])


def _even_flags(K, N, lda, ldb, fused: bool):
    """(EVEN_K, EVEN_N) for the masked kernels, from the uniform shapes.

    `stride(0) % 16 == 0` is required because EVEN_* also enables a
    `tl.multiple_of(..., [16, 16])` alignment hint; an unaligned stride would
    make that hint a lie and produce wrong results. For the fused kernel,
    EVEN_N tests the output width half = N // 2, not B's N.
    """
    even_k = (K % MAX_BLOCK_K == 0) and (lda % 16 == 0)
    if fused:
        even_n = (N % 2 == 0) and ((N // 2) % MAX_BLOCK_N == 0) and (ldb % 16 == 0)
    else:
        even_n = (N % MAX_BLOCK_N == 0) and (ldb % 16 == 0)
    return even_k, even_n


def moe_forward(hidden_states, topk_weights, topk_ids, weights: MoEWeights):
    """Single MoE compute path. Shapes: hidden (T, H), topk_* (T, k).

    Allocates its workspace from the actual routed-row count `T * top_k` and
    runs the two grouped GEMMs. Works unchanged under CUDA graph capture: the
    allocations then come from the graph's private memory pool.
    """
    num_tokens = hidden_states.shape[0]
    top_k = topk_ids.shape[1]
    assert hidden_states.dtype == weights.dtype, (
        f"activations are {hidden_states.dtype} but the weights are "
        f"{weights.dtype}; they must match at every GEMM")
    assert topk_weights.shape == topk_ids.shape == (num_tokens, top_k), (
        f"topk shapes {tuple(topk_weights.shape)} / {tuple(topk_ids.shape)} "
        f"do not match ({num_tokens}, {top_k})")
    assert topk_ids.dtype == torch.int64, f"topk_ids must be int64, got {topk_ids.dtype}"

    hidden_size = weights.hidden_size
    inter = weights.intermediate_size
    num_experts = weights.num_experts
    dtype = hidden_states.dtype
    device = hidden_states.device

    ws = _alloc_workspace(num_tokens, top_k, hidden_size, inter,
                          num_experts, dtype, device)
    _align_into(ws, hidden_states, topk_weights, topk_ids, top_k)

    even_k, even_n = _even_flags(hidden_size, 2 * inter, hidden_size, 2 * inter,
                                 fused=True)
    grouped_matmul_silu_kernel_masked[(lambda META: (META['NUM_SM'],))](
        ws.permuted, weights.w1_b_ptrs, ws.act,
        ws.counts, ws.row_offsets,
        hidden_size, 2 * inter, inter,
        num_experts,
        K=hidden_size, N=2 * inter,
        EVEN_K=even_k, EVEN_N=even_n,
        DTYPE=_triton_dtype(dtype),
    )

    even_k, even_n = _even_flags(inter, hidden_size, inter, hidden_size,
                                 fused=False)
    grouped_matmul_kernel_masked[(lambda META: (META['NUM_SM'],))](
        ws.act, weights.w2_b_ptrs, ws.down,
        ws.counts, ws.row_offsets,
        inter, hidden_size, hidden_size,
        num_experts,
        K=inter, N=hidden_size,
        EVEN_K=even_k, EVEN_N=even_n,
        DTYPE=_triton_dtype(dtype),
    )

    return _unpermute_into(ws, num_tokens, top_k)
