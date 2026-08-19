"""Triton kernels for the MoE FFN pair used by the zero-A2A EP path.

The zero-A2A path computes every aligned BLOCK_M-row chunk of the padded
token buffer with a single fused kernel: gate/up linear (fp32 accumulate),
SiLU activation, down projection, and the router-weight multiply.

The kernel relies on a sentinel discipline from the host side:

  - e_align[row] == e + 1  →  row belongs to local expert e.  Padding rows
    inside an expert block carry the same sentinel but have zero x/w data,
    so partial BLOCK_M chunks remain "uniform".
  - e_align[row] == 0      →  null / padding block (non-local tokens, unused
    capacity).  The kernel writes zeros for these rows.

A chunk whose rows disagree on the sentinel (or are all null) writes zeros
and returns early — required so that CUDA Graph replays never leave stale
data in the output buffer.

Design rationale (planer裁决 K1-K6):
  - Triton 3.1.0 requires all non-batch tl.dot dimensions >= 16, so the
    grid's first axis is T_ALIGN // BLOCK_M (64-token chunks) rather than
    individual (t,k) pairs.
  - Gate/up + down are fused in a single M-block loop (BLOCK_M=64, 12 iters
    @ M=768) to avoid a global y-scratch buffer, tl.debug_barrier hazards,
    and extra memory traffic.
  - Weights are aligned into fixed-capacity expert blocks (CAP) — all shapes
    depend only on N → graph-capturable.  Overflow tokens use vLLM-style
    capacity-drop semantics (negligible probability per Chernoff bound).
  - Router-weight multiply (MUL_ROUTED_WEIGHT) is done in fp32 before the
    final bf16 store, matching the vLLM requirement (vllm-project#31962).
  - If the compiler reports register spills or shared-memory exhaustion with
    num_warps=4, the accepted fallbacks are: reduce BLOCK_K to 64, reduce
    num_stages to 2, or split g/u computation into two sequential passes.
"""

import torch
import triton
import triton.language as tl

# --------------- constants ---------------

BLOCK_M = 64
BLOCK_D = 128
BLOCK_K = 128
NUM_WARPS = 4
NUM_STAGES = 2          # Reduced from 3 to fit RTX 4080 SUPER 99KB shared memory


# --------------- Triton kernel ---------------

@triton.jit
def moe_ffn_pair_kernel(
    x_ptr,      # [T_ALIGN, D] bf16  — aligned token buffer (padding rows = 0)
    wg_ptr,     # [E_local, M, D] bf16 — stacked gate weights
    wu_ptr,     # [E_local, M, D] bf16 — stacked up weights
    wd_ptr,     # [E_local, D, M] bf16 — stacked down weights  (out-dim first)
    w_ptr,      # [T_ALIGN] bf16       — router weights (padding rows = 0)
    e_ptr,      # [T_ALIGN] int64      — expert sentinel: e+1 for local experts, 0 = null
    out_ptr,    # [T_ALIGN, D] bf16    — output (kernel writes *every* row)
    M: tl.constexpr,         # moe_intermediate_size  (e.g. 768 — non-power-of-2)
    D: tl.constexpr,         # hidden_size            (e.g. 2048)
    BLOCK_M: tl.constexpr,   # 64  — rows / gate-up-out / down-K block size
    BLOCK_D: tl.constexpr,   # 128 — output (D) block size
    BLOCK_K: tl.constexpr,   # 128 — gate/up K (D) block size
):
    pid_t = tl.program_id(0)
    pid_d = tl.program_id(1)

    rows = pid_t * BLOCK_M + tl.arange(0, BLOCK_M)         # [BLOCK_M]
    d0 = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)            # [BLOCK_D]

    # ----- sentinel check: whole chunk must be one non-null expert -----
    e_vals = tl.load(e_ptr + rows)                          # [BLOCK_M] int64
    e_max = tl.max(e_vals, axis=0)
    e_min = tl.min(e_vals, axis=0)

    if e_min != e_max:
        # Mixed chunk (block boundaries misaligned — should never happen
        # with the host-side CAP % BLOCK_M == 0 guarantee).  Zero-fill so
        # graph replays never expose stale data from a previous invocation.
        zeros = tl.zeros([BLOCK_M, BLOCK_D], dtype=out_ptr.dtype.element_ty)
        tl.store(out_ptr + rows[:, None] * D + d0[None, :], zeros)
        return

    if e_max == 0:
        # Null / padding chunk — no local expert assigned.
        zeros = tl.zeros([BLOCK_M, BLOCK_D], dtype=out_ptr.dtype.element_ty)
        tl.store(out_ptr + rows[:, None] * D + d0[None, :], zeros)
        return

    e0 = e_max - 1           # true local expert id  (e_align = e+1 for valid rows)

    # compute base pointers for this expert (one-time)
    wg_base = wg_ptr + e0 * M * D
    wu_base = wu_ptr + e0 * M * D
    wd_base = wd_ptr + e0 * D * M

    # ----- fused FFN: gate/up → SiLU → down (single M-block outer loop) -----
    # res accumulates across M blocks (down projection)
    res = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    for m0 in range(0, M, BLOCK_M):           # 12 iterations @ M=768 / BLOCK_M=64
        m_off = m0 + tl.arange(0, BLOCK_M)     # [BLOCK_M]
        m_mask = m_off < M                     # [BLOCK_M] — non-power-of-2 guard (768)

        # --- gate / up: dot each K-block, accumulate in fp32 ---
        g = tl.zeros([BLOCK_M, BLOCK_M], dtype=tl.float32)
        u = tl.zeros([BLOCK_M, BLOCK_M], dtype=tl.float32)

        for d1 in range(0, D, BLOCK_K):       # 16 iterations @ D=2048 / BLOCK_K=128
            dk = d1 + tl.arange(0, BLOCK_K)    # [BLOCK_K]

            # x: [BLOCK_M, BLOCK_K] bf16  (re-read every m0 — L2 cached after
            # first m0 iteration; registers are too tight to hold all of D)
            x = tl.load(x_ptr + rows[:, None] * D + dk[None, :])

            # wg_t, wu_t: [BLOCK_K, BLOCK_M] bf16
            # offset = e0*M*D + dk[:, None]*M + m_off[None, :]
            wg_t = tl.load(wg_base + dk[:, None] * M + m_off[None, :],
                           mask=m_mask[None, :], other=0.)
            wu_t = tl.load(wu_base + dk[:, None] * M + m_off[None, :],
                           mask=m_mask[None, :], other=0.)

            g = tl.dot(x, wg_t, acc=g)         # fp32 accumulate
            u = tl.dot(x, wu_t, acc=u)

        # --- SiLU activation (fp32): y = silu(gate) * up ---
        y = tl.sigmoid(g) * g * u              # [BLOCK_M, BLOCK_M] fp32

        # --- down projection: [BLOCK_M, BD] = y @ Wd[m_block, :] ---
        # wd layout [E_local, D, M] → wd_t[i,j] = Wd[e0, d0+j, m0+i]
        # offset = e0*D*M + d0[None,:]*M + m_off[:,None]
        wd_t = tl.load(wd_base + d0[None, :] * M + m_off[:, None],
                       mask=m_mask[:, None], other=0.)   # [BLOCK_M, BLOCK_D]

        res = tl.dot(y.to(tl.bfloat16), wd_t, acc=res)   # [BLOCK_M, BLOCK_D] fp32

    # ----- router-weight multiply (MUL_ROUTED_WEIGHT) in fp32 -----
    w_row = tl.load(w_ptr + rows)                        # [BLOCK_M] bf16
    res = res * w_row.to(tl.float32)[:, None]             # broadcast over D

    # ----- store -----
    tl.store(out_ptr + rows[:, None] * D + d0[None, :],
             res.to(tl.bfloat16))


# --------------- host wrapper ---------------

# --------------- alignment kernel (replaces Python scatter loop) ---------------

@triton.jit
def moe_align_kernel(
    x_ptr,           # [NK, D] bf16   — flat token hidden states
    w_ptr,           # [NK] bf16      — flat routing weights
    is_local_ptr,    # [NK] int32     — 1 if expert is on this rank
    local_exp_ptr,   # [NK] int64     — local expert id (0..E_LOCAL-1), invalid for non-local
    pos_ptr,         # [NK] int64     — output: absolute row in x_align/w_align for gather
    keep_ptr,        # [NK] int32     — output: 1 = valid pair, 0 = non-local or overflow
    cnt_ptr,         # [E_LOCAL+1] int32 — atomic slot counters (pre-zeroed)
    x_align_ptr,     # [T_MAX, D] bf16 — output: aligned token buffer
    w_align_ptr,     # [T_MAX] bf16    — output: aligned weight buffer
    E_LOCAL: tl.constexpr,
    CAP: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Align (token, topk) pairs into fixed-capacity expert blocks.

    Each program handles one (token, topk) pair.  Local pairs atomically
    allocate a slot in their expert's block; non-local pairs return early
    (contribute zero through the keep mask later).
    """
    pid = tl.program_id(0)                     # 0 .. NK-1
    is_local = tl.load(is_local_ptr + pid)
    if is_local == 0:
        # Non-local pair → zero contribution; store safe defaults
        tl.store(pos_ptr + pid, 0)
        tl.store(keep_ptr + pid, 0)
        return

    e = tl.load(local_exp_ptr + pid)            # 0 .. E_LOCAL-1
    block_id = e + 1                            # 1 .. E_LOCAL  (0 = null block)

    # ---- atomic slot allocation within the expert's block ----
    slot = tl.atomic_add(cnt_ptr + block_id, 1)

    if slot >= CAP:
        # Overflow (vanishingly rare per Chernoff bound) → skip this pair
        tl.store(pos_ptr + pid, 0)
        tl.store(keep_ptr + pid, 0)
        return

    base_row = block_id * CAP + slot            # absolute row in align buffers
    tl.store(pos_ptr + pid, base_row)
    tl.store(keep_ptr + pid, 1)

    # ---- scatter token data ([1, D] → aligned row [1, D]) ----
    x_base = x_ptr + pid * D
    x_out_base = x_align_ptr + base_row * D
    for d in range(0, D, BLOCK_D):
        offs = d + tl.arange(0, BLOCK_D)
        x = tl.load(x_base + offs)
        tl.store(x_out_base + offs, x)

    # ---- scatter routing weight ----
    w = tl.load(w_ptr + pid)
    tl.store(w_align_ptr + base_row, w)


def moe_align(x_flat, w_flat, is_local, local_exp, x_align, w_align, cnt,
              E_LOCAL, CAP, BLOCK_D=128, num_warps=4, num_stages=2):
    """Align (token,topk) pairs into fixed-capacity expert blocks.

    Single Triton kernel replaces the Python scatter-loop (~100 launches).

    Args:
        x_flat:     [NK, D] bf16  — flat token hidden states
        w_flat:     [NK] bf16     — flat routing weights
        is_local:   [NK] int32    — 1 if expert is on this rank
        local_exp:  [NK] int64    — local expert id (0..E_LOCAL-1; invalid for non-local)
        x_align:    [T_MAX, D] bf16 — output buffer (must be pre-zeroed)
        w_align:    [T_MAX] bf16    — output buffer (must be pre-zeroed)
        cnt:        [E_LOCAL+1] int32 — atomic counters (must be pre-zeroed)
        E_LOCAL:    int           — number of local experts
        CAP:        int           — capacity slots per expert block

    Returns:
        pos:  [NK] int64 — absolute row index in align buffers (for inverse-gather)
        keep: [NK] int32 — 1 = valid pair, 0 = non-local or overflow
    """
    NK, D = x_flat.shape
    T_MAX = (E_LOCAL + 1) * CAP
    assert T_MAX == x_align.shape[0], f"T_MAX mismatch: {T_MAX} vs {x_align.shape[0]}"
    assert D % BLOCK_D == 0, f"D ({D}) must be a multiple of BLOCK_D ({BLOCK_D})"

    pos = torch.empty(NK, dtype=torch.int64, device=x_flat.device)
    keep = torch.empty(NK, dtype=torch.int32, device=x_flat.device)

    grid = (NK,)
    moe_align_kernel[grid](
        x_flat, w_flat, is_local, local_exp,
        pos, keep, cnt,
        x_align, w_align,
        E_LOCAL=E_LOCAL, CAP=CAP, D=D, BLOCK_D=BLOCK_D,
        num_warps=num_warps, num_stages=num_stages,
    )
    return pos, keep


# --------------- fused FFN kernel host wrapper ---------------

def moe_ffn_pair(x_align, wg, wu, wd, w_align, e_align, out_align,
                 M, D, BLOCK_M=64, BLOCK_D=128, BLOCK_K=128,
                 num_warps=4, num_stages=2):
    """Launch the fused MoE FFN pair kernel.

    Args:
        x_align:   [T_ALIGN, D] bf16  — aligned token buffer
        wg/wu/wd:  stacked expert weights [E_local, M, D] / [E_local, D, M] bf16
        w_align:   [T_ALIGN] bf16     — aligned router weights
        e_align:   [T_ALIGN] int64    — expert sentinel (0 = null)
        out_align: [T_ALIGN, D] bf16  — output (written in-place)
        M:         int  — moe_intermediate_size
        D:         int  — hidden_size
    """
    T_ALIGN = x_align.shape[0]
    assert T_ALIGN % BLOCK_M == 0, \
        f"T_ALIGN ({T_ALIGN}) must be a multiple of BLOCK_M ({BLOCK_M})"
    assert D % BLOCK_D == 0, \
        f"D ({D}) must be a multiple of BLOCK_D ({BLOCK_D})"

    grid = (T_ALIGN // BLOCK_M, D // BLOCK_D)
    moe_ffn_pair_kernel[grid](
        x_align, wg, wu, wd, w_align, e_align, out_align,
        M, D, BLOCK_M, BLOCK_D, BLOCK_K,
        num_warps=num_warps, num_stages=num_stages,
    )
