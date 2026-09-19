"""
MoE Grouped GEMM
================

A self-contained Mixture-of-Experts feed-forward module: routing, token
permutation, grouped GEMMs, and the un-permutation scatter -- plus a
CUDA-graph-ready variant of the whole flow.

Everything here is CUDA-only and works in fp16 or bf16. It is meant to be
dropped into a framework as a leaf module: it owns no global state, takes plain
tensors, and returns plain tensors.

What it computes
----------------

    hidden_states (T, H)
        |
        |-- route()               logits -> softmax -> top-k -> renormalize
        |                         -> topk_weights (T, k), topk_ids (T, k)
        |
        |-- permute_by_expert()   stable-sort the T*k (token, expert) pairs by
        |                         expert -> permuted (T*k, H), counts (E,)
        |
        |-- expert compute, grouped over the E experts:
        |      gate_up:  (M_i, H) x (H, 2I) -> (M_i, 2I)    [grouped GEMM]
        |      silu&mul: silu(gate) * up                    [fused into epilogue]
        |      down:     (M_i, I) x (I, H)  -> (M_i, H)     [grouped GEMM]
        |
        |-- unpermute_by_token()  scatter-add back to (T, H), weighted by routing

M_i = counts[i] is the number of tokens routed to expert i. It is an arbitrary
runtime value, which makes this a *grouped* GEMM with a jagged M -- see "The
M-direction mask" below.

Quick start
-----------

    import torch
    from moe_gemm import MoEWeights, route, moe_forward

    E, H, I, T, k = 128, 2048, 768, 512, 8
    w1 = torch.randn(E, 2 * I, H, device="cuda", dtype=torch.float16)
    w2 = torch.randn(E, H, I,     device="cuda", dtype=torch.float16)
    router = torch.randn(H, E,    device="cuda", dtype=torch.float16) * H ** -0.5

    weights = MoEWeights(w1, w2)      # built once, at model construction

    hidden = torch.randn(T, H, device="cuda", dtype=torch.float16)
    topk_weights, topk_ids = route(hidden, router, k)
    out = moe_forward(hidden, topk_weights, topk_ids, weights)          # (T, H)

CUDA-graph-ready path
---------------------

Most decode paths capture the whole forward pass in a CUDA graph. For that, use
`MoEGraphBuffers` instead of `moe_forward`. It pre-allocates every intermediate
buffer and keeps the per-expert pointer tables **on the device**, so the flow
contains no host round-trips and no host-computed addresses -- the two things
that make a capture either fail outright or, worse, succeed and silently
produce wrong results on replay.

    bufs = MoEGraphBuffers(weights, max_tokens=512, top_k=8)   # built once

    # per step: copy the new inputs into the fixed buffers, then run
    h, w, ids = bufs.set_inputs(hidden, topk_weights, topk_ids)
    out = moe_forward_graph(bufs, h, w, ids)

    # and to capture -- note what is inside the capture region and what is not:
    #     h, w, ids = bufs.set_inputs(hidden, w, ids)   # OUTSIDE
    #     with torch.cuda.graph(g):
    #         moe_forward_graph(bufs, h, w, ids)        # INSIDE
    #     bufs.set_inputs(hidden2, w2, ids2)            # OUTSIDE, before replay
    #     g.replay()

Both of those placement rules are enforced with assertions rather than left to
the reader: set_inputs() refuses to run during a capture, and moe_forward_graph()
refuses tensors that are not the views set_inputs() returned. Getting either
wrong produces no error on its own -- just replays that quietly ignore the new
data -- which is exactly the kind of mistake a comment does not prevent.

Inside a larger captured model, two more things change -- see
MoEGraphBufferPool and moe_forward_graph_captured. All MoE layers share ONE pool
of buffers (they execute in order; a per-layer set would waste ~48x the memory),
and the layer inputs are produced *inside* the capture region (the previous
layer's attention output), so they cannot go through set_inputs(). The captured
entry point asserts it is being called inside a capture -- the mirror image of
set_inputs()'s assertion -- and lets the layer pass graph-owned tensors straight
through.

Measured on an H=2048, I=768, E=128 (Qwen3-30B-A3B-like) shape at batch 1, the
graph-ready path is 3.1x faster end to end than the eager one (0.61ms -> 0.20ms
per layer), because at decode scale the eager path is CPU-launch-bound rather
than GPU-bound. At large batch the two converge, since the CPU overhead is then
hidden behind GPU execution.

Backends
--------

`backend="triton"` (default) uses the fused kernels below. `backend="torch_mm"`
uses torch._grouped_mm, a private PyTorch API (bf16, SM80+). The Triton path is
generally faster and is the only one with a graph-ready variant. Both assume
every expert has the same weight shapes.

The M-direction mask
--------------------

The upstream NVIDIA kernel assumes full tiles ("assume full tile for now") and
has no bounds checks: it will read and, worse, *write* out of bounds whenever
M_i is not a multiple of BLOCK_SIZE_M. MoE makes that the normal case.

The fix is a mask, but the cost of one depends on *which axis it cuts*:

  - `mask_am[:, None]` varies along rows but is uniform along A's contiguous
    (k) axis, so every row is either fully on or fully off -> the load can
    still be vectorized. Cheap.
  - `mask_k[None, :]` and `mask_bn[None, :]` vary along the contiguous axis,
    which forces element-wise predication and degrades the load to scalar.
    Expensive.

So the K- and N-direction masks are hoisted into the compile-time flags EVEN_K /
EVEN_N and eliminated entirely whenever the *model* dimensions (H, I) are
aligned -- which they always are in practice. Only the M-direction mask, the one
that is genuinely dynamic, is left at runtime.

Layout conventions
------------------

    w1, w2              (E, out, in)   nn.Linear layout, as loaded from a checkpoint
    weights.w1[e]       (K, N) contiguous  ->  (H, 2I) and (I, H)
    weights.stacked_w1() (E, K, N) bf16, for torch._grouped_mm
    hidden_states       (T, H)     fp16 or bf16, matching the weights
    topk_weights        (T, k)     same dtype as hidden_states
    topk_ids            (T, k)     int64
    permuted / M rows   (T*k, *)   same dtype, grouped by expert

Numerics
--------

    The Triton path supports fp16 and bf16. Pass weights and activations in the
    same dtype and everything downstream follows. Accumulation is fp32 in both
    cases, so switching to bf16 changes the storage precision only, not the
    arithmetic.

    torch._grouped_mm always works in bf16 (TORCH_MM_DTYPE), so the torch_mm
    backend casts to it regardless of what the input dtype was.

Requirements
------------

    torch >= 2.4, triton >= 3.0, CUDA SM80+.


Quantization (fp8 / fp4) -- what would have to change
-----------------------------------------------------

Not implemented; the notes are here because the current code contains a few
assumptions that are cheap to work around now and expensive to retrofit later.

The short version: **this file's structure survives quantization, but the kernel
bodies do not.** What is genuinely reusable is the scheduling skeleton, the mask
analysis, and the graph-ready buffer/table design -- none of which care about the
element type.

    layer                              per-tensor fp8   blockwise fp8   fp4 / MX
    ---------------------------------------------------------------------------
    route / permute / unpermute             unchanged       unchanged   unchanged
    graph-ready buffers + tables            unchanged    +1 scale table   +1 table
    virtual-SM scheduling skeleton           unchanged       unchanged   unchanged
    K-loop body                              unchanged   + dequantise   + unpack
    EVEN_K predicate                     min K = 32       same         MX block
    fp32 accumulator                         unchanged       unchanged   unchanged
    kernel ABI                              5 tables        6 tables     6 tables

The hard part is the **scales**, not the dtype:

  * per-tensor, or per-token(A)/per-channel(B): the output tile's factor is
    s_a[m] * s_b[n], both known at tile level, so it folds into the epilogue and
    the K loop is untouched. Cheap.
  * blockwise (128x128, DeepSeek-style) or MX (per 32 elements along K): the
    scale varies *within* the K loop, so each K block has to be dequantised
    before its tl.dot. That is a structural change, not an extra multiply, and it
    adds a scale pointer table to the ABI.

fp4 additionally breaks the "one element size" assumption: sub-byte packing
means A's leading dimension is measured in bytes while C's is in elements. That
is the same class of bug as the byte-vs-element trap noted in section 7 -- except
that now there is no single element_size to multiply by, because A, B and C
disagree. Newer Triton provides `tl.dot_scaled` for MX formats (check what your
version offers), which removes the hand-written unpacking but not the K-loop
rewrite.

Three seams are worth knowing about before touching any of this. Today the "5
device tables" ABI is assumed in three separate places -- `_even_flags` (takes
shapes), `_grouped_gemm`/`_grouped_gemm_silu` (build them), and
`_GroupedGemmStage` (fill them) -- so adding a scale table means editing all
three. And `MoEWeights` assumes a weight is a contiguous (K, N) fp16/bf16 list;
quantized weights are not that.

"""

from typing import Optional, Sequence

import torch
import torch.nn.functional as F

import triton
import triton.language as tl

__all__ = [
    "MoEWeights", "MoEGraphBufferPool", "MoEGraphBuffers",
    "moe_forward", "moe_forward_graph", "moe_forward_graph_captured",
    "route", "permute_by_expert", "unpermute_by_token",
    "torch_grouped_mm", "supports_torch_grouped_mm",
    "grouped_matmul_kernel_masked", "grouped_matmul_silu_kernel_masked",
    "BACKENDS", "DEVICE",
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
# vector) are equivalent for the two. A quantized dtype would not be -- see the
# "Quantization" section of the module docstring.
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


def _is_capturing() -> bool:
    """True while a CUDA graph capture is in flight on the current stream.

    Used to turn a documented-but-easy-to-violate rule ("copy the inputs in
    outside the capture region") into a hard failure. Violating it does not
    raise on its own -- it silently makes every replay re-load the capture-time
    input -- so a runtime check is worth more than a comment here.
    """
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


def _check_uniform(dtypes, what: str) -> torch.dtype:
    """All operands of a GEMM must share one element type (today, at least:
    a kernel with a quantized A and an fp16 C would need separate dtype
    parameters -- see the module docstring)."""
    unique = set(dtypes)
    assert len(unique) == 1, f"{what} must all share one dtype, got {sorted(unique)}"
    dtype = unique.pop()
    _triton_dtype(dtype)                      # raises on anything unsupported
    return dtype


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
    group_a_ptrs, group_b_ptrs, group_c_ptrs,
    # [group_size, 3] device tensor of <M, N, K> per gemm
    group_gemm_sizes,
    # [group_size, 3] device tensor of <lda, ldb, ldc> per gemm
    g_lds,
    group_size,
    NUM_SM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    EVEN_N: tl.constexpr,
    # Element type of A, B and C. Specialising on it (rather than hard-coding
    # fp16) is what lets the same kernel serve fp16 and bf16 -- both are 2-byte
    # MMA input types and both accumulate in fp32. It is also the seam a
    # quantized variant would widen into separate A/B/C dtypes.
    DTYPE: tl.constexpr,
):
    tile_idx = tl.program_id(0)
    last_problem_end = 0
    for g in range(group_size):
        gm = tl.load(group_gemm_sizes + g * 3)
        gn = tl.load(group_gemm_sizes + g * 3 + 1)
        gk = tl.load(group_gemm_sizes + g * 3 + 2)
        num_m_tiles = tl.cdiv(gm, BLOCK_SIZE_M)
        num_n_tiles = tl.cdiv(gn, BLOCK_SIZE_N)
        num_tiles = num_m_tiles * num_n_tiles
        while (tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles):
            k = gk
            lda = tl.load(g_lds + g * 3)
            ldb = tl.load(g_lds + g * 3 + 1)
            ldc = tl.load(g_lds + g * 3 + 2)
            a_ptr = tl.load(group_a_ptrs + g).to(tl.pointer_type(DTYPE))
            b_ptr = tl.load(group_b_ptrs + g).to(tl.pointer_type(DTYPE))
            c_ptr = tl.load(group_c_ptrs + g).to(tl.pointer_type(DTYPE))

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
            mask_bn = offs_bn < gn
            a_ptrs = a_ptr + offs_am[:, None] * lda + offs_k[None, :]
            b_ptrs = b_ptr + offs_k[:, None] * ldb + offs_bn[None, :]
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            for kk in range(0, tl.cdiv(k, BLOCK_SIZE_K)):
                if EVEN_K:
                    # K aligned -> drop the K mask entirely, and restore the
                    # alignment hint that lets Triton vectorize the load.
                    tl.multiple_of(a_ptrs, [16, 16])
                    a = tl.load(a_ptrs, mask=mask_am[:, None], other=0.0)
                    if EVEN_N:
                        tl.multiple_of(b_ptrs, [16, 16])
                        b = tl.load(b_ptrs)
                    else:
                        b = tl.load(b_ptrs, mask=mask_bn[None, :], other=0.0)
                else:
                    mask_k = offs_k < gk - kk * BLOCK_SIZE_K
                    a = tl.load(a_ptrs, mask=mask_am[:, None] & mask_k[None, :], other=0.0)
                    if EVEN_N:
                        b = tl.load(b_ptrs, mask=mask_k[:, None], other=0.0)
                    else:
                        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_bn[None, :], other=0.0)
                accumulator += tl.dot(a, b)
                a_ptrs += BLOCK_SIZE_K
                b_ptrs += BLOCK_SIZE_K * ldb
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
                tl.store(c_ptrs, c, mask=(offs_cm < gm)[:, None] & (offs_cn < gn)[None, :])

            tile_idx += NUM_SM

        last_problem_end = last_problem_end + num_tiles


# -----------------------------------------------------------------------------
# Fused epilogue: gate_up GEMM + SiLU, in one kernel
# -----------------------------------------------------------------------------
#
# The standard MoE flow splits the gate_up output's N axis (2I) into gate and up
# and then computes silu(gate) * up:
#
#     gate_up = A @ W1                                     (M, 2I)
#     act     = silu(gate_up[:, :I]) * gate_up[:, I:]      (M, I)
#
# The (M, 2I) intermediate is pure pass-through: written once, read once, then
# dead. Fusing it into the epilogue removes that round trip entirely.
#
# The catch is that gate and up occupy disjoint halves of the N axis, while one
# GEMM tile covers BLOCK_SIZE_N consecutive columns -- so an output tile
# [n0, n0+BN) needs two non-adjacent slices of gate_up, namely [n0, n0+BN) and
# [n0+I, n0+I+BN). The kernel therefore keeps two accumulators and loads both B
# slices in the K loop:
#
#     for kk in range(K / BK):
#         a  = load(A tile)                        # A is read once per tile
#         bg = load(B[:, n0      : n0+BN  ])       # gate columns
#         bu = load(B[:, n0+half : n0+half+BN])    # up columns
#         acc_g += tl.dot(a, bg)
#         acc_u += tl.dot(a, bu)
#     c = silu(acc_g) * acc_u                      # on the fp32 accumulators
#
# Two side effects worth knowing:
#
#   * The output N is I instead of 2I, so the tile count halves and **A is read
#     half as many times**. A is (M, K) and, in the shapes this targets, larger
#     than L2 -- so that saving is real HBM traffic and can exceed the SiLU
#     saving itself.
#   * Two (BM, BN) fp32 accumulators double the register pressure. At 128x128
#     that is 32768 registers, i.e. 256/thread at 4 warps -- over the 255
#     hardware limit, so it must run at 8 warps (hence the num_warps=8 configs).
#
# Accuracy: SiLU here runs on the fp32 accumulators, whereas the unfused path
# rounds gate_up to fp16 first. The fused version is therefore *more* accurate,
# not less.
#
# B's N must be exactly 2I. The kernel splits at half = N // 2; that is a MoE
# convention, not a general GEMM transform.

@triton.autotune(configs=_SILU_GEMM_CONFIGS, key=['group_size'])
@triton.jit
def grouped_matmul_silu_kernel_masked(
    group_a_ptrs, group_b_ptrs, group_c_ptrs,
    # [group_size, 3] device tensor of <M, N, K> per gemm.
    # NOTE: N here is B's N = 2I, not the output width. The output is N // 2.
    group_gemm_sizes,
    # [group_size, 3] device tensor of <lda, ldb, ldc> per gemm
    g_lds,
    group_size,
    NUM_SM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    # EVEN_N here tests the **output** width half = N // 2, because the tiles are
    # laid out over the output, not over B's N.
    EVEN_K: tl.constexpr,
    EVEN_N: tl.constexpr,
    # Element type of A, B and C. Specialising on it (rather than hard-coding
    # fp16) is what lets the same kernel serve fp16 and bf16 -- both are 2-byte
    # MMA input types and both accumulate in fp32. It is also the seam a
    # quantized variant would widen into separate A/B/C dtypes.
    DTYPE: tl.constexpr,
):
    tile_idx = tl.program_id(0)
    last_problem_end = 0
    for g in range(group_size):
        gm = tl.load(group_gemm_sizes + g * 3)
        gn = tl.load(group_gemm_sizes + g * 3 + 1)      # = 2I
        gk = tl.load(group_gemm_sizes + g * 3 + 2)
        half = gn // 2
        num_m_tiles = tl.cdiv(gm, BLOCK_SIZE_M)
        num_n_tiles = tl.cdiv(half, BLOCK_SIZE_N)
        num_tiles = num_m_tiles * num_n_tiles
        while (tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles):
            k = gk
            lda = tl.load(g_lds + g * 3)
            ldb = tl.load(g_lds + g * 3 + 1)
            ldc = tl.load(g_lds + g * 3 + 2)
            a_ptr = tl.load(group_a_ptrs + g).to(tl.pointer_type(DTYPE))
            b_ptr = tl.load(group_b_ptrs + g).to(tl.pointer_type(DTYPE))
            c_ptr = tl.load(group_c_ptrs + g).to(tl.pointer_type(DTYPE))

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
            # guarantees the up columns stay below gn and never run out of range.
            bg_ptrs = b_ptr + offs_k[:, None] * ldb + offs_bn[None, :]
            bu_ptrs = b_ptr + offs_k[:, None] * ldb + (offs_bn + half)[None, :]

            acc_g = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            acc_u = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            for kk in range(0, tl.cdiv(k, BLOCK_SIZE_K)):
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
                    mask_k = offs_k < gk - kk * BLOCK_SIZE_K
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
# 3. Host-side launch wrappers
# =============================================================================

def _even_flags(a_shapes, b_shapes, fused: bool):
    """Compute the (EVEN_K, EVEN_N) compile-time flags for the masked kernels.

    True only if **every** group qualifies. The `stride(0) % 16 == 0` part is
    required because EVEN_* also enables a `tl.multiple_of(..., [16, 16])`
    alignment hint on the load, which is what vectorization depends on: an
    unaligned stride would make that hint a lie and produce wrong results.

    For the fused kernel, EVEN_N tests half = N // 2 rather than B's N, since
    its tiles are laid out over the output. The two are not equivalent -- with
    N=128, half=64 satisfies neither `N % 128 == 0` nor `half % 128 == 0` in the
    same way -- so they cannot share one function.
    """
    even_k = all(k % MAX_BLOCK_K == 0 and ld % 16 == 0 for k, ld in a_shapes)
    if fused:
        even_n = all(n % 2 == 0 and (n // 2) % MAX_BLOCK_N == 0 and ld % 16 == 0
                     for n, ld in b_shapes)
    else:
        even_n = all(n % MAX_BLOCK_N == 0 and ld % 16 == 0 for n, ld in b_shapes)
    return even_k, even_n


def _shapes(group_a: Sequence[torch.Tensor], group_b: Sequence[torch.Tensor]):
    """(K, lda) per A and (N, ldb) per B -- what the even-flag predicates need."""
    return ([(a.shape[1], a.stride(0)) for a in group_a],
            [(b.shape[1], b.stride(0)) for b in group_b])


def _grouped_gemm(group_a, group_b) -> list:
    """grouped GEMM over the expert groups; returns a list of (M_i, N) tensors.

    This is the plain (unfused) kernel. It is used for the down projection, whose
    output width H has no gate/up structure to fuse.
    """
    assert len(group_a) == len(group_b)
    group_size = len(group_a)
    dtype = _check_uniform([t.dtype for t in group_a] + [t.dtype for t in group_b],
                           "grouped GEMM operands")
    even_k, even_n = _even_flags(*_shapes(group_a, group_b), fused=False)

    a_ptrs, b_ptrs, c_ptrs, sizes, lds, group_c = [], [], [], [], [], []
    for a, b in zip(group_a, group_b):
        assert a.shape[1] == b.shape[0], f"A K={a.shape[1]} but B K={b.shape[0]}"
        m, k = a.shape
        k, n = b.shape
        c = torch.empty((m, n), device=a.device, dtype=a.dtype)
        group_c.append(c)
        a_ptrs.append(a.data_ptr())
        b_ptrs.append(b.data_ptr())
        c_ptrs.append(c.data_ptr())
        sizes += [m, n, k]
        lds += [a.stride(0), b.stride(0), c.stride(0)]

    d_a = torch.tensor(a_ptrs, device=DEVICE)
    d_b = torch.tensor(b_ptrs, device=DEVICE)
    d_c = torch.tensor(c_ptrs, device=DEVICE)
    d_sizes = torch.tensor(sizes, dtype=torch.int32, device=DEVICE)
    d_lds = torch.tensor(lds, dtype=torch.int32, device=DEVICE)
    grouped_matmul_kernel_masked[(lambda META: (META['NUM_SM'],))](
        d_a, d_b, d_c, d_sizes, d_lds, group_size,
        EVEN_K=even_k, EVEN_N=even_n, DTYPE=_triton_dtype(dtype),
    )
    return group_c


def _grouped_gemm_silu(group_a, group_b) -> list:
    """Fused gate_up grouped GEMM + SiLU; returns a list of (M_i, N//2) tensors.

    Same arguments as _grouped_gemm (B's N is 2I) but the output width is I:

        out[:, j] = silu(gate_up[:, j]) * gate_up[:, j + I],  gate_up = A @ B

    The (M, 2I) intermediate never lands in memory.
    """
    assert len(group_a) == len(group_b)
    group_size = len(group_a)
    dtype = _check_uniform([t.dtype for t in group_a] + [t.dtype for t in group_b],
                           "grouped GEMM operands")
    even_k, even_n = _even_flags(*_shapes(group_a, group_b), fused=True)

    a_ptrs, b_ptrs, c_ptrs, sizes, lds, group_c = [], [], [], [], [], []
    for a, b in zip(group_a, group_b):
        assert a.shape[1] == b.shape[0], f"A K={a.shape[1]} but B K={b.shape[0]}"
        m, k = a.shape
        k, n = b.shape
        assert n % 2 == 0, f"gate_up N must be 2I (even), got {n}"
        # The only interface difference from _grouped_gemm: output width is N // 2
        c = torch.empty((m, n // 2), device=a.device, dtype=a.dtype)
        group_c.append(c)
        a_ptrs.append(a.data_ptr())
        b_ptrs.append(b.data_ptr())
        c_ptrs.append(c.data_ptr())
        sizes += [m, n, k]
        lds += [a.stride(0), b.stride(0), c.stride(0)]

    d_a = torch.tensor(a_ptrs, device=DEVICE)
    d_b = torch.tensor(b_ptrs, device=DEVICE)
    d_c = torch.tensor(c_ptrs, device=DEVICE)
    d_sizes = torch.tensor(sizes, dtype=torch.int32, device=DEVICE)
    d_lds = torch.tensor(lds, dtype=torch.int32, device=DEVICE)
    grouped_matmul_silu_kernel_masked[(lambda META: (META['NUM_SM'],))](
        d_a, d_b, d_c, d_sizes, d_lds, group_size,
        EVEN_K=even_k, EVEN_N=even_n, DTYPE=_triton_dtype(dtype),
    )
    return group_c


# =============================================================================
# 4. Routing, permutation, un-permutation
# =============================================================================

def route(hidden_states: torch.Tensor, router_weight: torch.Tensor, top_k: int):
    """Gating + top-k. Returns (topk_weights (T, k) fp16, topk_ids (T, k) int64).

    Note the fp32 upcast: it costs two fp32 temporaries per call (~(T*H + H*E)
    elements) and buys nothing -- router logits do not need it. Casting to fp16
    with an fp32 accumulate is generally safe here and cheaper; this stays fp32
    only because it is the numerically conservative default.
    """
    logits = hidden_states.float() @ router_weight.float()
    scores = logits.softmax(dim=-1)
    topk_weights, topk_ids = torch.topk(scores, top_k, dim=-1)
    # With k > 1 the selected weights are normally renormalized to sum to 1.
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_weights.to(hidden_states.dtype), topk_ids


def permute_by_expert(hidden_states, topk_ids, topk_weights, num_experts):
    """Sort the (token, expert) assignments by expert so each expert's rows are
    contiguous.

    Sorting position p corresponds to assignment a = order[p], i.e. the
    (order[p] % top_k)-th expert choice of token (order[p] // top_k).

    Returns:
        permuted:       (T*k, H)  activations, grouped by expert
        token_idx:      (T*k,)    source token index of each sorted position
        sorted_weights: (T*k,)    routing weight of each sorted position
        counts:         (E,)      tokens per expert -> the grouped GEMM's M_i
    """
    top_k = topk_ids.shape[1]
    flat_expert = topk_ids.reshape(-1)
    # stable sort: keeps tokens in their original order *within* an expert, so
    # the result is reproducible run to run
    order = torch.argsort(flat_expert, stable=True)
    token_idx = torch.div(order, top_k, rounding_mode="floor")
    permuted = hidden_states.index_select(0, token_idx).contiguous()
    sorted_weights = topk_weights.reshape(-1).index_select(0, order)
    counts = torch.bincount(flat_expert, minlength=num_experts)
    return permuted, token_idx, sorted_weights, counts


def unpermute_by_token(expert_out, token_idx, sorted_weights, num_tokens, out_dtype):
    """Scatter expert outputs back to (T, H), weighted and summed by routing.

    The accumulation is fp32 and only cast at the very end.
    """
    weights = sorted_weights.to(torch.float32).unsqueeze(-1)
    out = torch.zeros((num_tokens, expert_out.shape[-1]),
                      dtype=torch.float32, device=expert_out.device)
    out.index_add_(0, token_idx, expert_out.to(torch.float32) * weights)
    return out.to(out_dtype)


def _silu_and_mul_split(gate_up, intermediate_size):
    """Unfused silu(gate) * up over a per-expert list of (M_i, 2I) tensors.

    Only needed by the torch_mm backend; the Triton path fuses this into the
    GEMM epilogue.
    """
    out = []
    for x in gate_up:
        gate, up = x.split(intermediate_size, dim=-1)
        out.append(F.silu(gate) * up)
    return out


# =============================================================================
# 5. Weights
# =============================================================================

TORCH_MM_DTYPE = torch.bfloat16


def supports_torch_grouped_mm() -> bool:
    """torch._grouped_mm is a private API: CUDA SM80+ and a PyTorch that has it."""
    return (hasattr(torch, "_grouped_mm")
            and torch.cuda.is_available()
            and torch.cuda.get_device_capability()[0] >= 8)


class MoEWeights:
    """Expert weights, prepared once at model construction.

    Args:
        w1: (E, 2I, H) -- merged gate/up projection, nn.Linear layout, fp16/bf16.
        w2: (E, H, I)  -- down projection, nn.Linear layout, same dtype as w1.

    Both are stored transposed to (K, N) contiguous for the Triton kernels. The
    (E, K, N) bf16 stacks that torch._grouped_mm needs are built lazily on first
    use and cached, so a Triton-only run never pays for them.

    Per-expert weights must all have the same shape: this module stacks them and
    addresses experts by row offset, which requires a uniform layout.
    """

    def __init__(self, w1: torch.Tensor, w2: torch.Tensor):
        assert w1.dim() == 3 and w2.dim() == 3, "expected (E, out, in) tensors"
        assert w1.dtype == w2.dtype, \
            f"w1 and w2 must share one dtype, got {w1.dtype} / {w2.dtype}"
        self.dtype = w1.dtype
        _triton_dtype(self.dtype)         # reject anything the kernels cannot take
        num_experts, two_i, hidden = w1.shape
        assert w2.shape == (num_experts, hidden, two_i // 2), (
            f"w2 shape {tuple(w2.shape)} is not the (E, H, I) matching "
            f"w1 {tuple(w1.shape)}"
        )
        self.num_experts = num_experts
        self.hidden_size = hidden
        self.intermediate_size = two_i // 2
        # non-TMA kernels want B as (K, N) contiguous; the weights are fixed, so
        # this transpose should happen here and never inside the forward pass
        self.w1 = [w1[e].t().contiguous() for e in range(num_experts)]   # (H, 2I)
        self.w2 = [w2[e].t().contiguous() for e in range(num_experts)]   # (I, H)
        self._stacks: dict = {}

    def _stack(self, key: str, group_w) -> torch.Tensor:
        if key not in self._stacks:
            self._stacks[key] = torch.stack(group_w).to(TORCH_MM_DTYPE)
        return self._stacks[key]

    def stacked_w1(self) -> torch.Tensor:
        return self._stack("w1", self.w1)

    def stacked_w2(self) -> torch.Tensor:
        return self._stack("w2", self.w2)

    def __repr__(self):
        mb = sum(w.numel() * w.element_size() for w in self.w1 + self.w2) / 1e6
        return (f"MoEWeights(E={self.num_experts}, H={self.hidden_size}, "
                f"I={self.intermediate_size}, {mb:.0f}MB {self.dtype})")


def torch_grouped_mm(group_a, stacked_b: torch.Tensor) -> list:
    """torch._grouped_mm over a jagged set of groups.

    group_a[i] (M_i, K) is concatenated into one (sum M_i, K) tensor and the
    groups are delimited by `offs` (the inclusive cumulative row counts), so
    `stacked_b` must be the matching (E, K, N) bf16 stack -- i.e.
    MoEWeights.stacked_w1() / stacked_w2().

    torch._grouped_mm operates in bf16, hence the cast, and hence the wider
    tolerance any caller should use when comparing against an fp16 reference.
    """
    counts = torch.tensor([a.shape[0] for a in group_a], device=DEVICE,
                          dtype=torch.int32)
    offs = torch.cumsum(counts, 0).to(torch.int32)
    a = torch.cat(list(group_a), dim=0).to(TORCH_MM_DTYPE)
    out = torch._grouped_mm(a, stacked_b, offs=offs)
    return list(torch.split(out, counts.tolist(), dim=0))


# =============================================================================
# 6. Eager forward
# =============================================================================

BACKENDS = ("triton", "torch_mm")


def moe_forward(hidden_states: torch.Tensor,
                topk_weights: torch.Tensor,
                topk_ids: torch.Tensor,
                weights: MoEWeights,
                backend: str = "triton") -> torch.Tensor:
    """Eager MoE feed-forward. Shapes: hidden (T, H), topk_* (T, k). Returns (T, H).

    backend="triton"  fused gate_up+SiLU and a plain down GEMM (recommended)
    backend="torch_mm" torch._grouped_mm for both GEMMs, with an explicit
                       silu&mul in between (bf16, SM80+)

    For a CUDA-graph-captured path use MoEGraphBuffers / moe_forward_graph
    instead -- this function allocates per-expert tensors in a Python loop and
    must read back to the host for the token split, neither of which is
    capturable.
    """
    assert backend in BACKENDS, f"unknown backend {backend!r}, expected {BACKENDS}"
    assert hidden_states.dtype == weights.dtype, (
        f"activations are {hidden_states.dtype} but the weights are "
        f"{weights.dtype}; they must match at every GEMM"
    )
    num_tokens = hidden_states.shape[0]

    permuted, token_idx, sorted_weights, counts = permute_by_expert(
        hidden_states, topk_ids, topk_weights, weights.num_experts)
    group_a = list(torch.split(permuted, counts.tolist(), dim=0))

    if backend == "triton":
        # gate_up's SiLU is fused into the epilogue, so the (M, 2I) intermediate
        # never lands in memory. The down GEMM's output width H has no gate/up
        # structure, so it stays on the plain kernel.
        activated = _grouped_gemm_silu(group_a, weights.w1)
        expert_out = _grouped_gemm(activated, weights.w2)
    else:
        assert supports_torch_grouped_mm(), \
            "torch._grouped_mm needs CUDA SM80+ and a PyTorch that provides it"
        gate_up = torch_grouped_mm(group_a, weights.stacked_w1())
        activated = _silu_and_mul_split(gate_up, weights.intermediate_size)
        expert_out = torch_grouped_mm(activated, weights.stacked_w2())

    return unpermute_by_token(
        torch.cat(expert_out, dim=0), token_idx, sorted_weights,
        num_tokens, hidden_states.dtype,
    )


# =============================================================================
# 7. CUDA-graph-ready forward
# =============================================================================
#
# Why this exists: the eager path above cannot go inside a CUDA graph, for two
# independent reasons.
#
#   [G1] `counts.tolist()` is a device-to-host synchronisation. Any sync during
#        capture aborts the capture (cudaErrorStreamCaptureUnsupported).
#   [G2] `torch.tensor(python_list, device=...)` records the **addresses as they
#        were at capture time**. On replay the M_i values have changed, so the
#        per-expert pointer table points at the wrong rows. This one does not
#        raise -- it silently produces wrong results, which is worse.
#
# Both are fixed by the same move: keep every buffer pre-allocated and every
# address on the device. The per-expert start row is
#
#     row_offset[g] = sum(counts[:g])
#
# which torch.cumsum already computes on the device -- the eager path just threw
# that away and asked the host for counts instead. With row_offset on the device,
# the pointer table is plain device arithmetic:
#
#     a_ptrs[g] = a_base + row_offset[g] * (lda * element_size)
#
# Note the element_size: the table holds byte addresses while the leading
# dimension is in elements. Getting that wrong is a silent, plausible-looking
# error rather than a crash.
#
# One further trap: `torch.bincount` *looks* like a pure device op but computes
# its output length from `max(input)`, i.e. via `.item()`. It aborts a capture
# just like tolist() does. The scatter-add below is the capturable equivalent;
# counts are integers and fp32 represents them exactly below 2^24, so the two
# agree bit for bit.


class _GroupedGemmStage:
    """The five device tables one grouped GEMM needs, for a fixed pair of buffers.

    The lifetime is deliberately split in two:

      __init__  -- once, at model construction. Everything here depends only on
                   the weight shapes and the buffer addresses, both of which are
                   fixed for the life of the object, so the result can be baked
                   into a captured graph and never revisited.
      refresh   -- every forward. Refreshes only what depends on routing, and is
                   entirely device-side: no host value is involved.
    """

    def __init__(self, group_w, a_buf, c_buf, kernel, silu: bool):
        shapes = {tuple(w.shape) for w in group_w}
        assert len(shapes) == 1, \
            f"grouped path needs uniform weight shapes, got {sorted(shapes)}"
        num_experts = len(group_w)
        k, n = group_w[0].shape

        self.kernel = kernel
        self.num_experts = num_experts
        # A, B and C share one dtype today; a quantized variant would widen this
        # into per-operand dtypes (see the module docstring)
        _check_uniform([a_buf.dtype, c_buf.dtype, group_w[0].dtype],
                       "grouped GEMM operands")
        self.dtype = _triton_dtype(a_buf.dtype)
        self.lda = a_buf.stride(0)
        self.ldc = c_buf.stride(0)
        assert a_buf.shape[1] == k, f"A has K={a_buf.shape[1]}, weights have {k}"
        assert c_buf.shape[1] == (n // 2 if silu else n), "C width does not match"
        assert a_buf.is_contiguous() and c_buf.is_contiguous(), \
            "the grouped path addresses rows by offset, so both buffers must be contiguous"

        # Reuse the same predicates as the eager path rather than duplicating the
        # alignment rules: if the two drifted apart, the grouped path would
        # silently take the "drop the mask" branch when it should not, which is
        # an out-of-bounds store. A 0-row view carries the shape/stride the
        # predicates inspect.
        shapes_ = _shapes([a_buf[:0]], [group_w[0]])
        self.even_k, self.even_n = _even_flags(*shapes_, fused=silu)

        dev = a_buf.device
        # Byte strides: the pointer tables hold addresses.
        self.a_stride = self.lda * a_buf.element_size()
        self.c_stride = self.ldc * c_buf.element_size()

        # --- static: written once, never updated ---
        self.b_ptrs = torch.tensor([w.data_ptr() for w in group_w], device=dev)
        self.lds = torch.tensor([self.lda, n, self.ldc] * num_experts,
                                dtype=torch.int32, device=dev)
        self.sizes = torch.tensor([0, n, k] * num_experts,
                                  dtype=torch.int32, device=dev)   # M_i filled by refresh

        # --- dynamic ---
        self.a_ptrs = torch.empty((num_experts,), dtype=torch.int64, device=dev)
        self.c_ptrs = torch.empty((num_experts,), dtype=torch.int64, device=dev)
        # Base addresses as (E,) constants rather than a Python int, so that
        # refresh stays pure tensor arithmetic -- a host scalar in the expression
        # would get baked into the graph.
        self.a_base = torch.full((num_experts,), a_buf.data_ptr(),
                                 dtype=torch.int64, device=dev)
        self.c_base = torch.full((num_experts,), c_buf.data_ptr(),
                                 dtype=torch.int64, device=dev)

    def refresh(self, counts, row_offsets):
        """Update the three routing-dependent entries. Device-side only.

        `row_offsets` must be the **exclusive** prefix sum: row_offsets[g] =
        sum(counts[:g]), i.e. expert g's first row.
        """
        o = row_offsets.to(torch.int64)
        self.a_ptrs.copy_(self.a_base + o * self.a_stride)
        self.c_ptrs.copy_(self.c_base + o * self.c_stride)
        self.sizes[0::3].copy_(counts)      # only M_i varies; N and K are static

    def launch(self):
        self.kernel[(lambda META: (META['NUM_SM'],))](
            self.a_ptrs, self.b_ptrs, self.c_ptrs, self.sizes, self.lds,
            self.num_experts, EVEN_K=self.even_k, EVEN_N=self.even_n,
            DTYPE=self.dtype,
        )


class MoEGraphBufferPool:
    """整模型共享的 MoE 中间 buffer（跨层一份）。

    为什么需要：CUDA graph 录制 kernel 及其访问的地址；图内 MoE 的中间张量
    必须预分配、地址固定，否则 replay 会静默读写 capture 时的旧地址（[G2]）。
    48 层各持一份 MoEGraphBuffers（A3B、max_tokens=512 时 ≈74MB）合计 ≈3.5GB；
    整模型 graph 中各层顺序执行，因此一套 buffer 足够。

    为什么与通信无关：a2a 已弃用；AR[EP] 直接原地读写 final 张量，不需要任何
    额外收发 buffer。eager 路径（moe_forward）也不使用本类。
    """

    def __init__(self, dtype: torch.dtype, num_experts: int, hidden_size: int,
                 intermediate_size: int, max_tokens: int, top_k: int):
        _triton_dtype(dtype)
        self.dtype = dtype
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_tokens = max_tokens
        self.top_k = top_k
        self.max_rows = max_tokens * top_k

        e = num_experts
        hidden, inter, rows = hidden_size, intermediate_size, self.max_rows

        def buf(*shape, dt=dtype):
            return torch.empty(shape, dtype=dt, device=DEVICE)

        # 输入暂存（set_inputs / staged 路径用；图内路径不经过它们）
        self.in_hidden = buf(max_tokens, hidden)
        self.in_w = buf(max_tokens, top_k)
        self.in_ids = torch.empty((max_tokens, top_k), dtype=torch.int64, device=DEVICE)

        # intermediates and output
        self.permuted = buf(rows, hidden)       # permute output = gate_up's A
        self.act = buf(rows, inter)             # gate_up+SiLU output = down's A
        self.down = buf(rows, hidden)           # down output = unpermute's input
        self.final = buf(max_tokens, hidden)

        # routing-dependent state, all device-resident
        self.counts = torch.empty((e,), dtype=torch.int32, device=DEVICE)
        self.row_offsets = torch.empty((e,), dtype=torch.int32, device=DEVICE)
        self.token_idx = torch.empty((rows,), dtype=torch.int64, device=DEVICE)
        self.sorted_weights = buf(rows)
        # fp32 counter for the histogram; see moe_forward_graph on why this is
        # not torch.bincount
        self.count_f = torch.zeros((e,), dtype=torch.float32, device=DEVICE)
        self.ones_f = torch.ones((rows,), dtype=torch.float32, device=DEVICE)

        # unpermute scratch
        self.scratch = buf(rows, hidden, dt=torch.float32)
        self.acc = buf(max_tokens, hidden, dt=torch.float32)

    def check_compatible(self, weights: MoEWeights, top_k: int,
                         max_tokens: int) -> None:
        """一个 pool 只能服务同 (E, H, I, dtype) 的层——跨层共享的前提。"""
        assert (weights.num_experts == self.num_experts
                and weights.hidden_size == self.hidden_size
                and weights.intermediate_size == self.intermediate_size), (
            f"pool(E={self.num_experts}, H={self.hidden_size}, I={self.intermediate_size}) "
            f"不能服务 weights(E={weights.num_experts}, H={weights.hidden_size}, "
            f"I={weights.intermediate_size})"
        )
        assert weights.dtype == self.dtype, (
            f"weights dtype {weights.dtype} != pool dtype {self.dtype}")
        assert top_k == self.top_k, f"top_k {top_k} != pool 的 {self.top_k}"
        assert max_tokens <= self.max_tokens, \
            f"max_tokens {max_tokens} 超过 pool 容量 {self.max_tokens}"

    def nbytes(self) -> int:
        return sum(v.numel() * v.element_size()
                   for v in self.__dict__.values() if isinstance(v, torch.Tensor))


class MoEGraphBuffers:
    """固定 buffer + 静态表，供一层 MoE 的 graph 路径使用。

    buffer 可来自共享的 MoEGraphBufferPool（整模型一份，48 层顺序执行互不
    冲突）；不传 pool 时自建（单层测试/独立使用）。每层独有的是
    _GroupedGemmStage 的两张指针表（b_ptrs 指向该层权重、sizes 的 M_i 逐层
    不同，合计 KB 级）。

    Three rules, all of them hard requirements of CUDA graph capture:

    1. **Inputs must be copied into in_hidden / in_w / in_ids.** A graph captures
       addresses, so passing an external tensor bakes *that* tensor's address into
       the graph; a later step passing a different tensor object will still read
       the old address. Use set_inputs().

    2. **The return value is a view into the (shared) final buffer**, overwritten
       by the next call -- including the next LAYER's forward when the pool is
       shared. Clone or copy it if it needs to outlive the call; consume it in
       layer order otherwise.

    3. **Only one forward may be in flight per pool.** The row offsets and the
       permute buffers are shared state, so two interleaved forwards would clobber
       each other. Layer-by-layer execution -- which the layer structure gives you
       -- is exactly the supported pattern.

    Capacity is max_tokens * top_k permuted rows.
    """

    # 与旧实现同名的池化 buffer；_permute_into/_unpermute_into 通过这些属性
    # 访问，别名化后无需改动。
    _POOL_ATTRS = ("in_hidden", "in_w", "in_ids", "permuted", "act", "down",
                   "final", "counts", "row_offsets", "token_idx",
                   "sorted_weights", "count_f", "ones_f", "scratch", "acc")

    def __init__(self, weights: MoEWeights, max_tokens: Optional[int] = None,
                 top_k: Optional[int] = None, dtype: Optional[torch.dtype] = None,
                 pool: Optional[MoEGraphBufferPool] = None):
        # default to the weights' dtype: the activations must match the weights
        # at every GEMM, so letting the two differ is only ever a mistake
        dtype = weights.dtype if dtype is None else dtype
        _triton_dtype(dtype)
        assert dtype == weights.dtype, (
            f"activation dtype {dtype} must match the weights' {weights.dtype}"
        )
        self.dtype = dtype
        e = weights.num_experts
        hidden, inter = weights.hidden_size, weights.intermediate_size

        if pool is None:
            assert max_tokens is not None and top_k is not None, \
                "不传 pool 时必须给出 max_tokens 与 top_k"
            pool = MoEGraphBufferPool(dtype, e, hidden, inter, max_tokens, top_k)
        else:
            pool.check_compatible(
                weights,
                top_k if top_k is not None else pool.top_k,
                max_tokens if max_tokens is not None else pool.max_tokens,
            )
        self.pool = pool
        self.num_experts = e
        self.top_k = pool.top_k
        self.max_tokens = pool.max_tokens
        self.max_rows = pool.max_rows
        self.hidden_size = hidden
        self.intermediate_size = inter
        for name in self._POOL_ATTRS:
            setattr(self, name, getattr(pool, name))

        self.stage_gu = _GroupedGemmStage(weights.w1, self.permuted, self.act,
                                          grouped_matmul_silu_kernel_masked, silu=True)
        self.stage_d = _GroupedGemmStage(weights.w2, self.act, self.down,
                                         grouped_matmul_kernel_masked, silu=False)

    def set_inputs(self, hidden_states, topk_weights, topk_ids):
        """Copy this step's inputs into the fixed buffers; returns their views.

        Must be called **outside** the captured region: a copy performed during
        capture bakes in its source address too, so replay would just move the
        capture-time input over whatever you loaded since. That failure is
        silent -- the replay simply ignores your new data -- so it is asserted
        against rather than merely documented.
        """
        assert not _is_capturing(), (
            "set_inputs() was called inside a CUDA graph capture. The copy would "
            "be recorded together with its source address, so every replay would "
            "move the capture-time input instead of the new one. Call set_inputs() "
            "outside the capture region and pass the views it returns to "
            "moe_forward_graph() inside it."
        )
        t = hidden_states.shape[0]
        assert t <= self.max_tokens, f"batch {t} exceeds capacity {self.max_tokens}"
        assert topk_ids.shape[1] == self.top_k, f"top_k must stay {self.top_k}"
        self.in_hidden[:t].copy_(hidden_states)
        self.in_w[:t].copy_(topk_weights)
        self.in_ids[:t].copy_(topk_ids)
        return self.in_hidden[:t], self.in_w[:t], self.in_ids[:t]

    def nbytes(self) -> int:
        return sum(v.numel() * v.element_size()
                   for v in self.__dict__.values() if isinstance(v, torch.Tensor))


def _permute_into(bufs: MoEGraphBuffers, hidden_states, topk_weights, topk_ids):
    """Stage 1: permutation + histogram + exclusive prefix sum. All device-side."""
    top_k = bufs.top_k
    n = hidden_states.shape[0] * top_k

    flat = topk_ids.reshape(-1)
    order = torch.argsort(flat, stable=True)
    bufs.token_idx[:n].copy_(torch.div(order, top_k, rounding_mode="floor"))
    bufs.sorted_weights[:n].copy_(topk_weights.reshape(-1).index_select(0, order))
    torch.index_select(hidden_states, 0, bufs.token_idx[:n], out=bufs.permuted[:n])

    # torch.bincount would abort a graph capture (see the section note above).
    bufs.count_f.zero_()
    bufs.count_f.scatter_add_(0, flat, bufs.ones_f[:n])
    bufs.counts.copy_(bufs.count_f)                  # fp32 -> int32, exact
    bufs.row_offsets.copy_(torch.cumsum(bufs.counts, 0) - bufs.counts)


def _unpermute_into(bufs: MoEGraphBuffers, num_tokens: int):
    """Stage 4: weighted scatter-add back to (T, H), into a fixed buffer."""
    n = num_tokens * bufs.top_k
    w = bufs.sorted_weights[:n].to(torch.float32).unsqueeze(-1)
    torch.mul(bufs.down[:n], w, out=bufs.scratch[:n])
    bufs.acc[:num_tokens].zero_()
    bufs.acc[:num_tokens].index_add_(0, bufs.token_idx[:n], bufs.scratch[:n])
    return bufs.final[:num_tokens].copy_(bufs.acc[:num_tokens])


def _graph_forward_impl(bufs: MoEGraphBuffers, hidden_states, topk_weights, topk_ids):
    """capture 兼容的 MoE 全流程（permute → 两个 grouped GEMM → unpermute）。

    调用方负责保证三个输入的地址跨 replay 稳定——图外请用 set_inputs()，
    图内直接传图私有内存池里的张量。
    """
    num_tokens = hidden_states.shape[0]

    _permute_into(bufs, hidden_states, topk_weights, topk_ids)

    bufs.stage_gu.refresh(bufs.counts, bufs.row_offsets)
    bufs.stage_gu.launch()
    bufs.stage_d.refresh(bufs.counts, bufs.row_offsets)
    bufs.stage_d.launch()

    return _unpermute_into(bufs, num_tokens)


def moe_forward_graph(bufs: MoEGraphBuffers, hidden_states, topk_weights, topk_ids):
    """Graph-capturable MoE forward. Bit-compatible with moe_forward(backend="triton").

    The three tensor arguments must be the views returned by bufs.set_inputs().

    Measured on H=2048, I=768, E=128, batch=1: 0.20ms/layer captured vs 0.61ms
    eager (3.1x). At batch 512 the gap closes to 1.06x, because by then the GPU
    has enough work to hide the host-side launch cost that capture removes.
    """
    # A graph captures addresses. Capturing with an external tensor bakes *that*
    # tensor's address into the graph, and a later set_inputs() would then update
    # a buffer the graph never reads -- silently, with no error until someone
    # notices the answers are stale. Requiring the buffer views makes that
    # misuse impossible rather than merely documented.
    assert (hidden_states.data_ptr() == bufs.in_hidden.data_ptr()
            and topk_weights.data_ptr() == bufs.in_w.data_ptr()
            and topk_ids.data_ptr() == bufs.in_ids.data_ptr()), (
        "the three inputs must be the views returned by bufs.set_inputs(), so "
        "that the graph reads from the buffers set_inputs() writes to"
    )
    assert hidden_states.dtype == bufs.dtype, (
        f"activations are {hidden_states.dtype} but this buffer set was built for "
        f"{bufs.dtype}"
    )
    return _graph_forward_impl(bufs, hidden_states, topk_weights, topk_ids)


def moe_forward_graph_captured(bufs: MoEGraphBuffers, hidden_states,
                               topk_weights, topk_ids):
    """图内入口：三个输入产生于 capture 区内（图私有内存池，地址跨 replay 稳定）。

    与 set_inputs() 的关系是镜像的——后者断言"不在 capture 内"，这里断言
    "在 capture 内"。两者存在的理由相同：把"图外/图内走错分支"从静默错误
    （replay 读到 capture 时的旧数据）变成硬失败。整模型 CUDA graph 里 MoE
    的输入来自上一层（attention 输出、路由结果），不可能经 set_inputs 暂存，
    这个入口就是那条路径。

    整模型集成时的用法：capture 区内 `moe_forward_graph_captured(bufs, x, w, ids)`，
    其中 x/w/ids 是图内张量；capture 外同一层走 set_inputs() + moe_forward_graph()。
    """
    assert _is_capturing(), (
        "moe_forward_graph_captured() 只在 capture 区内合法。图外调用会把"
        "外部张量的地址烘进图里（replay 静默读旧数据）；图外请用 "
        "bufs.set_inputs() + moe_forward_graph()"
    )
    assert hidden_states.shape[0] <= bufs.max_tokens, (
        f"batch {hidden_states.shape[0]} exceeds capacity {bufs.max_tokens}"
    )
    assert hidden_states.dtype == bufs.dtype, (
        f"activations are {hidden_states.dtype} but this buffer set was built for "
        f"{bufs.dtype}"
    )
    return _graph_forward_impl(bufs, hidden_states, topk_weights, topk_ids)