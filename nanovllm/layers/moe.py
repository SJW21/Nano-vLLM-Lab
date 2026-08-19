import torch
from torch import nn
import torch.nn.functional as F

from nanovllm.layers.linear import ReplicatedLinear
from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.moe_kernels import moe_ffn_pair, moe_align, BLOCK_M
from nanovllm.distributed.parallel_state import get_ep_group

# Module-level cache for alignment buffers (shared across all MoE layers
# because all layers have the same D and N → identical T_MAX).
_za2a_buf_cache: dict[tuple, dict[str, torch.Tensor]] = {}


def _get_or_create_za2a_buffers(device, dtype, N, D, T_MAX, E_LOCAL=0):
    """Get or create shared alignment buffers for the zero-A2A path.

    All MoE layers share the same buffers because they have the same D
    and N (hence identical T_MAX).  This cuts memory from O(L) to O(1).
    """
    key = (device.index if device.index is not None else device, N, D, T_MAX)
    if key not in _za2a_buf_cache:
        _za2a_buf_cache[key] = {
            "x_align":  torch.empty(T_MAX, D, dtype=dtype, device=device),
            "w_align":  torch.empty(T_MAX,    dtype=dtype, device=device),
            "e_align":  torch.zeros(T_MAX,    dtype=torch.int64, device=device),
            "cnt":      torch.zeros(max(E_LOCAL + 1, 1), dtype=torch.int32, device=device),
            "out_align": torch.empty(T_MAX, D, dtype=dtype, device=device),
            "partial":  torch.empty(N, D,     dtype=dtype, device=device),
        }
    return _za2a_buf_cache[key]


class ExpertFFN(nn.Module):
    """Individual expert FFN with full (unsharded) weights.

    In EP mode, each rank holds a subset of experts with complete weights.
    A single expert has small weight footprint (3 × hidden × moe_intermediate),
    so there is no need for TP sharding inside the expert.
    """

    def __init__(
        self,
        hidden_size: int,
        moe_intermediate_size: int,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, moe_intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, moe_intermediate_size, bias=False)
        self.down_proj = nn.Linear(moe_intermediate_size, hidden_size, bias=False)
        self.act = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, hidden_size]
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        # SiluAndMul expects concatenated [gate, up] along last dim
        y = self.act(torch.cat([gate, up], dim=-1))
        return self.down_proj(y)


class MoELayer(nn.Module):
    """Mixture-of-Experts layer supporting EP and TP parallel modes.

    EP mode (moe_mode="ep"):
        Four communication backends are available via `comm_mode`:

        "auto" (default):
            Uses all_to_all for eager execution (bandwidth-optimal, zero compute
            waste) and automatically switches to zero_a2a during CUDA Graph
            capture (fixed-shape, graph-compatible, zero token dispatch).

        "zero_a2a":
            Zero all-to-all: no token dispatch at all.  Each rank keeps its
            local tokens and computes partial expert outputs only for the
            experts it owns, then an AllReduce sums the partials.
            CUDA-Graph compatible (fixed-shape Triton kernel over padded
            aligned buffers) and cheaper than all_gather (no duplicate compute).
            Requires DP=1 (all ranks hold identical tokens).

        "all_gather":
            Always uses AllGather → slot-dispatch → ReduceScatter. CUDA-Graph
            compatible, but has compute waste from fixed-capacity buffers when
            running eagerly.

        "all_to_all":
            Always uses two-phase all-to-all with metadata exchange. Most
            bandwidth-efficient for eager mode, but dynamic split sizes
            preclude CUDA Graph capture.

    TP mode (moe_mode="tp"):
        Each rank owns all experts but with TP-sharded weights. Hidden states
        are all-gathered across ranks, each rank computes its shard, then
        results are reduce-scattered back.

    The router (gate) is always fully replicated (no sharding) since it is
    small: hidden_size × num_experts ≈ 2048 × 128 ≈ 262K parameters.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        moe_intermediate_size: int,
        num_experts: int,
        top_k: int,
        ep_size: int,
        ep_rank: int,
        moe_mode: str = "ep",
        comm_mode: str = "auto",
        hidden_act: str = "silu",
        norm_topk_prob: bool = True,
    ):
        super().__init__()
        assert hidden_act == "silu", "Only SiLU activation is supported for MoE experts."
        assert moe_mode in ("ep", "tp"), f"Invalid moe_mode: {moe_mode}"
        assert comm_mode in ("all_gather", "all_to_all", "auto", "zero_a2a"), \
            f"Invalid comm_mode: {comm_mode}"
        assert num_experts % ep_size == 0, \
            f"num_experts ({num_experts}) must be divisible by ep_size ({ep_size})"

        self.hidden_size = hidden_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.ep_size = ep_size
        self.ep_rank = ep_rank
        self.moe_mode = moe_mode
        self.comm_mode = comm_mode
        self.norm_topk_prob = norm_topk_prob
        self.experts_per_rank = num_experts // ep_size

        # Router (gate): fully replicated, small weight footprint
        # Named "gate" to match HF checkpoint key: model.layers.X.mlp.gate.weight
        self.gate = ReplicatedLinear(hidden_size, num_experts, bias=False)

        # Expert weights: when EP > 1, store as stacked [E, D, M] tensors
        # directly (ready for bmm, no runtime copies).  For ep_size == 1
        # (128 experts → would be ~3.6 GB/layer), keep individual ExpertFFN
        # modules and stack only active experts on-the-fly in _compute_local.
        if ep_size > 1:
            # --- Stacked parameters (EP mode, ~32 experts/rank) ---
            # Shape follows nn.Linear convention [out_features, in_features]:
            #   gate/up: [moe_intermediate, hidden]   = [M, D]
            #   down:    [hidden, moe_intermediate]    = [D, M]
            # bmm uses .transpose(1,2) to get [E, D, M] — a view, zero copy.
            E = self.experts_per_rank
            self.stacked_gate = nn.Parameter(
                torch.empty(E, moe_intermediate_size, hidden_size))
            self.stacked_up = nn.Parameter(
                torch.empty(E, moe_intermediate_size, hidden_size))
            self.stacked_down = nn.Parameter(
                torch.empty(E, hidden_size, moe_intermediate_size))

            # Custom weight-loaders: write directly into expert slot.
            def _make_loader():
                def loader(param, loaded_weight, expert_id=0):
                    param.data[expert_id].copy_(loaded_weight)
                return loader
            self.stacked_gate.weight_loader = _make_loader()
            self.stacked_up.weight_loader   = _make_loader()
            self.stacked_down.weight_loader = _make_loader()

            # Keep an empty experts list for API compatibility with tests.
            self.experts = nn.ModuleList()
        else:
            # --- Individual experts (ep_size=1, 128 experts/rank) ---
            self.experts = nn.ModuleList([
                ExpertFFN(hidden_size, moe_intermediate_size)
                for _ in range(self.experts_per_rank)
            ])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass for MoE layer.

        Args:
            hidden_states: [N, hidden_size] input token representations.

        Returns:
            [N, hidden_size] output after expert computation and merge.
        """
        # 1. Router
        scores = self.gate(hidden_states)               # [N, num_experts]
        weights, indices = self._topk_routing(scores)      # [N, top_k] each

        if self.moe_mode == "ep":
            if self.ep_size == 1:
                # No communication needed — use the efficient local path.
                return self._forward_ep_local(hidden_states, weights, indices)
            if self.comm_mode == "all_gather":
                return self._forward_ep_ag(hidden_states, weights, indices)
            elif self.comm_mode == "all_to_all":
                return self._forward_ep_ata(hidden_states, weights, indices)
            elif self.comm_mode == "zero_a2a":
                return self._forward_ep_za2a(hidden_states, weights, indices)
            else:  # "auto"
                # During CUDA Graph capture, use the graph-compatible zero_a2a
                # path; otherwise use the bandwidth-optimal all_to_all path.
                if torch.cuda.is_current_stream_capturing():
                    return self._forward_ep_za2a(hidden_states, weights, indices)
                else:
                    return self._forward_ep_ata(hidden_states, weights, indices)
        else:
            return self._forward_tp(hidden_states, weights, indices)

    # ------------------------------------------------------------------
    # EP mode (all_to_all): dispatch → compute → combine
    # ------------------------------------------------------------------

    def _forward_ep_ata(
        self,
        hidden_states: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        N, D = hidden_states.shape
        K = self.top_k
        ep_group = get_ep_group() if self.ep_size > 1 else None

        # --- Phase 0: route + EP mapping ---
        # ep_rank_target: which EP rank owns the expert for each (token, top_k) pair
        ep_rank_target = indices // self.experts_per_rank      # [N, K], range [0, ep_size)
        local_expert_id = indices % self.experts_per_rank       # [N, K], range [0, experts_per_rank)

        # --- Phase 0b: metadata exchange (skip if ep_size == 1) ---
        if self.ep_size > 1:
            send_counts = torch.bincount(
                ep_rank_target.reshape(-1), minlength=self.ep_size
            ).to(torch.int64)
            # Exchange split sizes (ep_size int64 values ~ 64 bytes total)
            recv_counts = torch.empty(self.ep_size, dtype=torch.int64, device=hidden_states.device)
            torch.distributed.all_to_all_single(
                recv_counts, send_counts,
                output_split_sizes=[1] * self.ep_size,
                input_split_sizes=[1] * self.ep_size,
                group=ep_group.device_group,
            )
            recv_counts = recv_counts.tolist()
            total_recv = sum(recv_counts)
        else:
            send_counts = [N * K]
            recv_counts = [N * K]
            total_recv = N * K
            # ep_rank_target is all zeros when ep_size=1

        # --- Phase 1: prepare dispatch tensors ---
        # Expand and flatten: [N, D] → [N, K, D] → [N*K, D]
        tokens_flat = hidden_states.unsqueeze(1).expand(-1, K, -1).reshape(-1, D)
        weights_flat = weights.reshape(-1)                       # [N*K]
        local_exp_flat = local_expert_id.reshape(-1)             # [N*K]
        ep_rank_flat = ep_rank_target.reshape(-1)                # [N*K]

        # --- Phase 2: sort + dispatch all-to-all ---
        if self.ep_size > 1:
            # Sort by target EP rank so tokens to the same destination are contiguous
            sort_idx = torch.argsort(ep_rank_flat)
            unsort_idx = torch.argsort(sort_idx)

            tokens_sorted = tokens_flat[sort_idx]
            weights_sorted = weights_flat[sort_idx]
            local_exp_sorted = local_exp_flat[sort_idx]

            # Dispatch tokens, weights, expert ids
            dispatched_tokens = ep_group.all_to_all(
                tokens_sorted, output_split_sizes=recv_counts, input_split_sizes=send_counts.tolist()
            )
            dispatched_weights = ep_group.all_to_all(
                weights_sorted, output_split_sizes=recv_counts, input_split_sizes=send_counts.tolist()
            )
            dispatched_exp_ids = ep_group.all_to_all(
                local_exp_sorted, output_split_sizes=recv_counts, input_split_sizes=send_counts.tolist()
            )
        else:
            dispatched_tokens = tokens_flat
            dispatched_weights = weights_flat
            dispatched_exp_ids = local_exp_flat
            unsort_idx = torch.arange(N * K, device=hidden_states.device)

        # --- Phase 3: local expert compute ---
        expert_outputs = self._compute_local(dispatched_tokens, dispatched_exp_ids)

        # --- Phase 4: combine all-to-all (reverse dispatch) ---
        if self.ep_size > 1:
            # Reverse: send split = recv_counts, receive split = send_counts
            combined_outputs = ep_group.all_to_all(
                expert_outputs, output_split_sizes=send_counts.tolist(), input_split_sizes=recv_counts
            )
            combined_weights = ep_group.all_to_all(
                dispatched_weights, output_split_sizes=send_counts.tolist(), input_split_sizes=recv_counts
            )
        else:
            combined_outputs = expert_outputs
            combined_weights = dispatched_weights

        # --- Phase 5: unsort + weighted merge ---
        combined_outputs = combined_outputs[unsort_idx]     # [N*K, D]
        combined_weights = combined_weights[unsort_idx]     # [N*K]

        output_3d = combined_outputs.reshape(N, K, D)
        weights_3d = combined_weights.reshape(N, K)
        output = (output_3d * weights_3d.unsqueeze(-1)).sum(dim=1)  # [N, D]

        return output

    # ------------------------------------------------------------------
    # EP mode (all_gather): AllGather → slot-dispatch → ReduceScatter
    #
    # CUDA-Graph-compatible: every tensor shape is determined solely by
    # the (fixed) batch size N.  Slot dispatch uses cumsum + index_add_
    # instead of nonzero/dynamic-indexing so the entire forward pass can
    # be captured by torch.cuda.CUDAGraph for decode-phase acceleration.
    # ------------------------------------------------------------------

    def _forward_ep_ag(
        self,
        hidden_states: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """AllGather-based EP forward — graph-capturable, no dynamic splits.

        Flow:
          1. AllGather hidden_states, weights, indices → all ranks see all tokens
          2. Fixed-capacity slot dispatch: cumsum assigns consecutive slots per
             expert; index_add_ copies tokens into fixed-size per-expert buffers.
          3. ExpertFFN on fixed-size buffers (graph-capturable matmuls).
          4. Gather results back + weighted accumulate.
          5. ReduceScatter to return local-token slices.
        """
        N, D = hidden_states.shape
        K = self.top_k

        # --- ep_size=1 shortcut (no communication) ---
        if self.ep_size == 1:
            return self._forward_ep_local(hidden_states, weights, indices)

        ep_group = get_ep_group()
        E = self.ep_size

        # --- Phase 1: AllGather ---
        # All ranks now hold the full token set [N*E, ...]
        global_hidden = ep_group.all_gather(hidden_states, dim=0)   # [N*E, D]
        global_weights = ep_group.all_gather(weights, dim=0)         # [N*E, K]
        global_indices = ep_group.all_gather(indices, dim=0)         # [N*E, K]

        N_all = N * E

        # --- Phase 2: expert routing (all fixed-shape elementwise ops) ---
        # global_indices are global expert IDs (0 .. num_experts-1)
        owner_rank = global_indices // self.experts_per_rank          # [N_all, K]
        local_exp_id = global_indices % self.experts_per_rank         # [N_all, K]
        is_local = (owner_rank == self.ep_rank)                       # [N_all, K]

        # Flatten to [N_all*K, ...] for slot dispatch
        hidden_flat = global_hidden.unsqueeze(1).expand(-1, K, -1).reshape(-1, D)
        weights_flat = global_weights.reshape(-1)                     # [N_all*K]
        is_local_flat = is_local.reshape(-1)                          # [N_all*K]
        local_exp_flat = local_exp_id.reshape(-1)                     # [N_all*K]

        N_flat = N_all * K

        # --- Phase 3: capacity calculation ---
        # Capacity: allocate enough slots per expert so overflow probability
        # is negligible.  Expected tokens per expert = N_flat / num_experts.
        # With capacity = 3× expected, Chernoff bound gives P(overflow) < 10⁻⁷⁴
        # for balanced routing in the decode regime (N ≤ 512).
        expected = max(N_flat // self.num_experts, 1)
        capacity = min(expected * 3, N_flat)   # never exceed total tokens
        capacity = max(capacity, 64)            # floor at 64 slots

        # --- Phase 3b: per-expert dispatch + compute ---
        # Per-expert matmul (not bmm): use [capacity, D] temporary per expert,
        # re-used across the loop.  Memory = capacity × D ≈ 1.5 MB vs bmm's
        # [E, capacity, D] ≈ 48 MB.  CUDA Graph eliminates per-iteration kernel
        # launch overhead, so bmm's fusion benefit is marginal here.
        output = torch.zeros(N_all, D, dtype=hidden_states.dtype,
                             device=hidden_states.device)
        E = self.experts_per_rank
        expert_buffer = torch.zeros(capacity, D,
                                    dtype=hidden_states.dtype,
                                    device=hidden_states.device)

        for e in range(E):                                   # fixed iteration count
            mask = is_local_flat & (local_exp_flat == e)     # [N_flat]
            slot_idx = torch.cumsum(mask.int(), dim=0) - 1   # [N_flat]
            safe_idx = slot_idx.clamp(0, capacity - 1)       # [N_flat]
            mask_float = mask.to(hidden_flat.dtype).unsqueeze(-1)

            # Fill per-expert buffer
            expert_buffer.zero_()
            expert_buffer.index_add_(0, safe_idx, hidden_flat * mask_float)

            # Per-expert FFN (uses stacked weight for fast lookup)
            gate = torch.mm(expert_buffer, self.stacked_gate[e].t())
            up   = torch.mm(expert_buffer, self.stacked_up[e].t())
            out_e = torch.mm(F.silu(gate) * up, self.stacked_down[e].t())
            # out_e: [capacity, D]

            # Gather results
            gathered = out_e[safe_idx] * mask_float          # [N_flat, D]
            w = weights_flat.unsqueeze(-1) * mask_float       # [N_flat, 1]
            gathered = gathered * w

            gathered_3d = gathered.reshape(N_all, K, D)
            output += gathered_3d.sum(dim=1)                  # [N_all, D]

        # --- Phase 4: ReduceScatter ---
        return ep_group.reduce_scatter(output, dim=0)         # [N, D]

    # ------------------------------------------------------------------
    # Shared local path (ep_size=1): no communication needed
    # ------------------------------------------------------------------

    def _forward_ep_local(
        self,
        hidden_states: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """EP forward when ep_size=1 — pure local computation, no collectives."""
        N, D = hidden_states.shape
        K = self.top_k

        tokens_flat = hidden_states.unsqueeze(1).expand(-1, K, -1).reshape(-1, D)
        weights_flat = weights.reshape(-1)
        expert_ids_flat = indices.reshape(-1)

        expert_outputs = self._compute_local(tokens_flat, expert_ids_flat)

        output_3d = expert_outputs.reshape(N, K, D)
        weights_3d = weights_flat.reshape(N, K)
        output = (output_3d * weights_3d.unsqueeze(-1)).sum(dim=1)
        return output

    # ------------------------------------------------------------------
    # TP mode: all-gather → compute → reduce-scatter
    # ------------------------------------------------------------------

    def _forward_tp(
        self,
        hidden_states: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        N, D = hidden_states.shape
        K = self.top_k
        ep_group = get_ep_group()

        # All-gather: share all tokens across EP ranks
        all_tokens = ep_group.all_gather(hidden_states, dim=0)  # [N*ep_size, D]

        # Each rank computes its TP shard for all experts × all tokens
        partial = self._compute_all_experts_tp(all_tokens, weights, indices)

        # Reduce-scatter: split results back to original ranks
        output = ep_group.reduce_scatter(partial, dim=0)  # [N, D]
        return output

    # ------------------------------------------------------------------
    # EP mode (zero_a2a): local compute + all_reduce combine
    #
    # Under DP=1 (all ranks hold identical tokens) there is no need to
    # dispatch tokens at all.  Each rank computes partial expert outputs
    # only for the experts it owns, writing zeros for non-local (t,k)
    # pairs.  A single AllReduce then sums the partials across EP ranks.
    #
    # Tokens are aligned into fixed-capacity expert blocks (CAP per
    # expert + 1 null block) so every tensor shape depends only on the
    # (fixed) batch size N → fully CUDA-Graph capturable.
    #
    # Buffers are shared across all MoE layers via a module-level cache
    # (_get_or_create_za2a_buffers) — all layers have identical D and
    # N, so alignment shapes are identical.
    # ------------------------------------------------------------------

    def _forward_ep_za2a(
        self,
        hidden_states: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """Zero-A2A EP forward — CUDA-Graph capturable, no token dispatch.

        Flow:
          1. Expand hidden_states to [N*K, D] and map each (token, topk)
             pair to its local expert id (or -1 for non-local).
          2. Scatter pairs into fixed-capacity per-expert blocks (one block
             per local expert + one null block for non-local tokens).
             Every tensor shape depends only on N → graph-safe.
          3. Single fused Triton kernel (gate/up → SiLU → down → route
             weight multiply) over the aligned buffer.
          4. Inverse-gather results back to original (t,k) order, merge
             across top-k, and AllReduce partials across EP ranks.
        """
        N, D = hidden_states.shape
        K = self.top_k

        if self.ep_size == 1:
            return self._forward_ep_local(hidden_states, weights, indices)
        if N == 0:
            return hidden_states

        # ---- threshold: small batch → bmm path (CAP=N, zero overflow) ----
        # torch.topk returns K *distinct* experts per token, so per-expert
        # load is bounded by N.  CAP = N guarantees no capacity drop.
        # For larger batches the Triton alignment path has better padding
        # efficiency (BLOCK_M-aligned CAP vs 4× padding of bmm path).
        if N * K <= 512:   # N ≤ 64 for K=8
            return self._forward_ep_za2a_bmm(hidden_states, weights, indices)

        E_local = self.experts_per_rank
        NK = N * K
        ep_group = get_ep_group()

        # ---- capacity: fixed-size blocks, graph-safe ----
        # Expected tokens per (global) expert = NK / num_experts.
        # CAP = ceil(3 × expected / BLOCK_M) × BLOCK_M, floor BLOCK_M.
        # Overflow probability is negligible per Chernoff bound (same
        # argument as the all_gather path).
        expected = (3 * NK + self.num_experts - 1) // self.num_experts
        CAP = max(((expected + BLOCK_M - 1) // BLOCK_M) * BLOCK_M, BLOCK_M)
        T_MAX = (E_local + 1) * CAP

        # Reuse buffers across layers: use module-level cache keyed by
        # (D, T_MAX) so all 48 MoE layers share the same alignment buffers.
        bufs = _get_or_create_za2a_buffers(hidden_states.device,
                                           hidden_states.dtype, N, D, T_MAX, E_local)

        # ---- 1. expand + route mapping (all fixed-shape) ----
        x_flat = hidden_states.unsqueeze(1).expand(-1, K, -1).reshape(NK, D)
        e_global = indices.reshape(NK)
        is_local_bool = (e_global // E_local) == self.ep_rank      # [NK] bool
        is_local = is_local_bool.to(torch.int32)                     # [NK] int32 for Triton
        local_exp = torch.where(is_local_bool, e_global % E_local, -1)  # [NK] int64
        w_flat = weights.reshape(NK)

        # ---- 2. alignment: 1 Triton kernel replaces ~100 scatter launches ----
        # Pre-zero the align buffers (padding rows remain zero → FFN kernel
        # computes on zero data, producing zero output for empty slots).
        bufs["x_align"].zero_()
        bufs["w_align"].zero_()
        bufs["cnt"].zero_()

        pos_flat, keep = moe_align(
            x_flat, w_flat, is_local, local_exp,
            bufs["x_align"], bufs["w_align"], bufs["cnt"],
            E_local, CAP,
        )

        # Fill e_align sentinels (once per buffer at creation time).
        # Null block = 0, expert e block = e+1.
        if not bufs.get("_e_init"):
            for e in range(E_local):
                bufs["e_align"][e * CAP:(e + 1) * CAP] = e + 1
            bufs["_e_init"] = True

        # ---- 3. fused Triton kernel (single launch) ----
        moe_ffn_pair(
            bufs["x_align"], self.stacked_gate, self.stacked_up, self.stacked_down,
            bufs["w_align"], bufs["e_align"], bufs["out_align"],
            M=self.moe_intermediate_size, D=D,
        )

        # ---- 4. inverse-gather + merge ----
        # pos_flat is the absolute row index (computed by the align kernel).
        # keep=0 for non-local and overflow pairs → zero contribution.
        keep_flat = keep.to(hidden_states.dtype)                     # [NK] bf16 (0. or 1.)
        out_flat = bufs["out_align"][pos_flat] * keep_flat.unsqueeze(-1)   # [NK, D]

        bufs["partial"].zero_()
        bufs["partial"].copy_(out_flat.reshape(N, K, D).sum(dim=1))   # [N, D]

        # ---- 5. zero-a2a combine: AllReduce partials across EP ranks ----
        ep_group.all_reduce(bufs["partial"])         # in-place
        return bufs["partial"]

    def _forward_ep_za2a_bmm(
        self,
        hidden_states: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """Zero-A2A decode-optimised path: torch.bmm instead of Triton kernel.

        Uses CAP = N (provably zero overflow — topk returns K distinct
        experts per token) and batched matmul across all local experts:

            x3d [E_local, CAP, D] @ W^T → gate/up [E_local, CAP, M]
            silu(gate)*up @ W_down → out [E_local, CAP, D]

        This eliminates the 95% padding waste of the BLOCK_M-aligned path
        for small N and leverages cuBLAS tensor cores.
        """
        N, D = hidden_states.shape
        K = self.top_k
        if self.ep_size == 1:
            return self._forward_ep_local(hidden_states, weights, indices)
        if N == 0:
            return hidden_states

        E_local = self.experts_per_rank
        NK = N * K
        ep_group = get_ep_group()

        # CAP = N: per-expert load ≤ N (each token contributes at most 1
        # pair to any expert because topk picks K *distinct* experts).
        CAP = N
        T_MAX = (E_local + 1) * CAP

        bufs = _get_or_create_za2a_buffers(hidden_states.device,
                                           hidden_states.dtype, N, D, T_MAX, E_local)

        # ---- 1. expand + route mapping ----
        x_flat = hidden_states.unsqueeze(1).expand(-1, K, -1).reshape(NK, D)
        e_global = indices.reshape(NK)
        is_local_bool = (e_global // E_local) == self.ep_rank      # [NK] bool
        is_local = is_local_bool.to(torch.int32)                     # [NK] int32 for Triton
        local_exp = torch.where(is_local_bool, e_global % E_local, -1)  # [NK] int64
        w_flat = weights.reshape(NK)

        # ---- 2. alignment: 1 Triton kernel replaces scatter loop ----
        # Note: w_align is written by moe_align but unused by bmm — harmless dead write.
        bufs["x_align"].zero_()
        bufs["cnt"].zero_()

        pos_flat, keep = moe_align(
            x_flat, w_flat, is_local, local_exp,
            bufs["x_align"], bufs["w_align"], bufs["cnt"],
            E_local, CAP,
        )

        # ---- 3. bmm FFN (replaces Triton kernel) ----
        # x3d: [E_local, CAP, D] — view of x_align, expert blocks 1..E_local
        x3d = bufs["x_align"].view(E_local + 1, CAP, D)[1:]        # [E, C, D]

        gate = torch.bmm(x3d, self.stacked_gate.transpose(1, 2))    # [E, C, M]
        up   = torch.bmm(x3d, self.stacked_up.transpose(1, 2))      # [E, C, M]
        y    = F.silu(gate) * up                                      # [E, C, M] bf16
        out3d = torch.bmm(y, self.stacked_down.transpose(1, 2))      # [E, C, D]

        # out3d_view writes into the [1:] slice of out_align
        out3d_view = bufs["out_align"].view(E_local + 1, CAP, D)[1:]
        out3d_view.copy_(out3d)

        # ---- 4. inverse-gather + merge (route weights applied here) ----
        # pos_flat is the absolute row index (computed by the align kernel).
        # keep=0 for non-local and overflow pairs → zero contribution.
        keep_flat = keep.to(hidden_states.dtype)
        out_flat = bufs["out_align"][pos_flat] * (w_flat * keep_flat).unsqueeze(-1)

        bufs["partial"].zero_()
        bufs["partial"].copy_(out_flat.reshape(N, K, D).sum(dim=1))

        # ---- 5. all-reduce combine ----
        ep_group.all_reduce(bufs["partial"])
        return bufs["partial"]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _topk_routing(self, scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Softmax + top-k routing.

        Args:
            scores: [N, num_experts] raw router logits.

        Returns:
            weights: [N, top_k] normalized routing weights.
            indices: [N, top_k] selected expert indices (long).
        """
        orig_dtype = scores.dtype
        scores = F.softmax(scores, dim=-1, dtype=torch.float32)
        weights, indices = torch.topk(scores, self.top_k, dim=-1)
        if self.norm_topk_prob:
            weights = weights / weights.sum(dim=-1, keepdim=True)
        return weights.to(orig_dtype), indices

    def _compute_local(
        self,
        tokens: torch.Tensor,
        expert_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Per-expert FFN compute — memory-efficient, no padding.

        Groups tokens by expert then iterates over active experts with
        individual matmuls.  Uses stacked weights (EP>1) for fast lookup;
        CUDA Graph eliminates per-expert launch overhead during decode.

        Args:
            tokens: [total_tokens, hidden_size] dispatched tokens.
            expert_ids: [total_tokens] local expert index (0..experts_per_rank-1).

        Returns:
            [total_tokens, hidden_size] expert outputs in original order.
        """
        total_tokens = tokens.size(0)
        if total_tokens == 0:
            return tokens

        D = self.hidden_size

        # --- Group tokens by local expert ---
        sort_idx = torch.argsort(expert_ids)
        tokens_sorted = tokens[sort_idx]
        ids_sorted = expert_ids[sort_idx]

        unique_exp, counts = torch.unique_consecutive(ids_sorted, return_counts=True)

        output_sorted = tokens_sorted.new_empty(total_tokens, D)

        if self.ep_size > 1:
            # EP mode: use stacked [E, out, in] parameters.
            ptr = 0
            for e, count in zip(unique_exp.tolist(), counts.tolist()):
                x = tokens_sorted[ptr:ptr + count]                # [cnt, D]
                gate = torch.mm(x, self.stacked_gate[e].t())      # [cnt, M]
                up   = torch.mm(x, self.stacked_up[e].t())        # [cnt, M]
                out  = torch.mm(F.silu(gate) * up,
                                self.stacked_down[e].t())         # [cnt, D]
                output_sorted[ptr:ptr + count] = out
                ptr += count
        else:
            # ep_size=1: use individual ExpertFFN modules.
            ptr = 0
            for e, count in zip(unique_exp.tolist(), counts.tolist()):
                x = tokens_sorted[ptr:ptr + count]
                output_sorted[ptr:ptr + count] = self.experts[e](x)
                ptr += count

        # --- Restore original order ---
        unsort_idx = torch.argsort(sort_idx)
        return output_sorted[unsort_idx]

    def _compute_all_experts_tp(
        self,
        all_tokens: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """TP-mode: compute all experts on all tokens (partial result per rank).

        Each rank holds a TP shard of every expert's weight. The result is a
        partial sum that will be reduced across ranks.

        Args:
            all_tokens: [total_tokens, hidden_size] all-gathered tokens.
            weights: [N, top_k] from the local batch (needed for merge).
            indices: [N, top_k] from the local batch.

        Returns:
            [total_tokens, hidden_size] partial output.
        """
        total_tokens = all_tokens.size(0)
        K = self.top_k
        D = self.hidden_size

        # For TP mode, each rank has all experts (with TP-sharded weights).
        # Compute expert outputs for all tokens.
        all_weights, all_indices = self._topk_routing(self.gate(all_tokens))

        # Flatten for per-expert compute
        tokens_flat = all_tokens.unsqueeze(1).expand(-1, K, -1).reshape(-1, D)
        exp_ids_flat = all_indices.reshape(-1)
        weights_flat = all_weights.reshape(-1)

        # Group by expert and compute
        expert_outputs = self._compute_local(tokens_flat, exp_ids_flat)

        # Weighted merge
        outputs_3d = expert_outputs.reshape(total_tokens, K, D)
        weights_3d = weights_flat.reshape(total_tokens, K)
        return (outputs_3d * weights_3d.unsqueeze(-1)).sum(dim=1)
