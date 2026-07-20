import torch
from torch import nn
import torch.nn.functional as F

from nanovllm.layers.linear import ReplicatedLinear
from nanovllm.layers.activation import SiluAndMul
from nanovllm.distributed.parallel_state import get_ep_group


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
        Experts are partitioned across ranks. Each rank owns num_experts/ep_size
        complete experts. Tokens are dispatched via all-to-all to the rank that
        owns the target expert, computed locally, then combined back.

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
        hidden_act: str = "silu",
        norm_topk_prob: bool = True,
    ):
        super().__init__()
        assert hidden_act == "silu", "Only SiLU activation is supported for MoE experts."
        assert moe_mode in ("ep", "tp"), f"Invalid moe_mode: {moe_mode}"
        assert num_experts % ep_size == 0, \
            f"num_experts ({num_experts}) must be divisible by ep_size ({ep_size})"

        self.hidden_size = hidden_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.ep_size = ep_size
        self.ep_rank = ep_rank
        self.moe_mode = moe_mode
        self.norm_topk_prob = norm_topk_prob
        self.experts_per_rank = num_experts // ep_size

        # Router (gate): fully replicated, small weight footprint
        # Named "gate" to match HF checkpoint key: model.layers.X.mlp.gate.weight
        self.gate = ReplicatedLinear(hidden_size, num_experts, bias=False)

        # Local experts: subset of all experts assigned to this rank
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
            return self._forward_ep(hidden_states, weights, indices)
        else:
            return self._forward_tp(hidden_states, weights, indices)

    # ------------------------------------------------------------------
    # EP mode: dispatch → compute → combine
    # ------------------------------------------------------------------

    def _forward_ep(
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
        """Compute expert FFN on dispatched tokens, grouped by expert.

        Args:
            tokens: [total_tokens, hidden_size] dispatched tokens.
            expert_ids: [total_tokens] local expert index (0..experts_per_rank-1).

        Returns:
            [total_tokens, hidden_size] expert outputs in original order.
        """
        if tokens.size(0) == 0:
            return tokens

        total_tokens = tokens.size(0)
        dtype = tokens.dtype

        # Group tokens by local expert
        sort_idx = torch.argsort(expert_ids)
        tokens_sorted = tokens[sort_idx]
        ids_sorted = expert_ids[sort_idx]

        unique_experts, counts = torch.unique_consecutive(ids_sorted, return_counts=True)

        output = torch.zeros(total_tokens, self.hidden_size, dtype=dtype, device=tokens.device)
        ptr = 0
        for exp_id, count in zip(unique_experts.tolist(), counts.tolist()):
            output[ptr:ptr + count] = self.experts[exp_id](tokens_sorted[ptr:ptr + count])
            ptr += count

        # Restore original order
        unsort_idx = torch.argsort(sort_idx)
        return output[unsort_idx]

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
