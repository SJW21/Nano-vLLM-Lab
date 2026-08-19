"""
Tests for the MoE layer (ExpertFFN, MoELayer in EP mode).

Run:
    python tests/test_moe.py                          # all tests
    python tests/test_moe.py ExpertFFN MoELayerEP1    # specific test groups

Requires: torch with CUDA or CPU-only (gloo backend for single-process tests).
"""

import os
import sys
import math
import torch
import torch.nn.init as init
import torch.distributed as dist
import torch.nn.functional as F

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanovllm.layers.moe import ExpertFFN, MoELayer
from nanovllm.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
    get_ep_group,
    reset_parallel_groups,
)


# ---------------------------------------------------------------------------
# Session-level setup / teardown
# ---------------------------------------------------------------------------

_world_size = 1
_backend = "gloo"  # gloo works on CPU; use "nccl" for multi-GPU tests


def session_setup():
    """Initialize distributed environment once for the entire test session."""
    if dist.is_initialized():
        return
    dist.init_process_group(
        backend=_backend,
        init_method="tcp://127.0.0.1:29500",
        world_size=_world_size,
        rank=0,
    )
    initialize_model_parallel(
        world_size=_world_size,
        tensor_parallel_size=1,
        data_parallel_size=1,
        expert_parallel_size=_world_size,
        backend=_backend,
    )


def session_teardown():
    """Clean up distributed environment once."""
    reset_parallel_groups()
    if dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _init_moe_gate_weights(moe: MoELayer):
    """Initialize the gate (ReplicatedLinear) weights.

    LinearBase subclasses (ReplicatedLinear, ColumnParallelLinear, etc.) do
    not call reset_parameters() — they rely on checkpoint loading to fill
    weights.  For unit tests without a checkpoint, we must init manually.
    """
    # Gate / router: standard Kaiming uniform (same as nn.Linear)
    init.kaiming_uniform_(moe.gate.weight, a=math.sqrt(5))


def assert_allclose(actual, expected, msg="", atol=1e-4, rtol=1e-4):
    """Assert with informative error message."""
    if not torch.allclose(actual, expected, atol=atol, rtol=rtol):
        diff = (actual - expected).abs().max().item()
        raise AssertionError(f"{msg} Max diff: {diff:.8f}")


def check_no_nan_inf(tensor, name="tensor"):
    """Check tensor has no NaN or Inf values."""
    if torch.isnan(tensor).any():
        raise AssertionError(f"{name} contains NaN values")
    if torch.isinf(tensor).any():
        raise AssertionError(f"{name} contains Inf values")


# ---------------------------------------------------------------------------
# Test: ExpertFFN
# ---------------------------------------------------------------------------

class TestExpertFFN:

    @classmethod
    def setup_class(cls):
        session_setup()
        torch.manual_seed(42)
        cls.hidden_size = 256
        cls.moe_intermediate = 128
        cls.ffn = ExpertFFN(cls.hidden_size, cls.moe_intermediate)
        cls.ffn.eval()

    def test_forward_shape(self):
        x = torch.randn(4, self.hidden_size)
        out = self.ffn(x)
        assert out.shape == (4, self.hidden_size), \
            f"Expected (4, {self.hidden_size}), got {out.shape}"

    def test_forward_deterministic(self):
        x = torch.randn(2, self.hidden_size)
        out1 = self.ffn(x)
        out2 = self.ffn(x)
        assert_allclose(out1, out2, "ExpertFFN should be deterministic")

    def test_no_nan(self):
        x = torch.randn(4, self.hidden_size)
        out = self.ffn(x)
        check_no_nan_inf(out, "ExpertFFN output")


# ---------------------------------------------------------------------------
# Test: MoELayer ep_size=1
# ---------------------------------------------------------------------------

class TestMoELayerEP1:

    HIDDEN = 512
    MOE_INTERMEDIATE = 256
    NUM_EXPERTS = 8
    TOP_K = 2

    @classmethod
    def setup_class(cls):
        session_setup()
        torch.manual_seed(42)
        cls.moe = MoELayer(
            hidden_size=cls.HIDDEN,
            intermediate_size=2048,
            moe_intermediate_size=cls.MOE_INTERMEDIATE,
            num_experts=cls.NUM_EXPERTS,
            top_k=cls.TOP_K,
            ep_size=1,
            ep_rank=0,
            moe_mode="ep",
            norm_topk_prob=True,
        )
        _init_moe_gate_weights(cls.moe)
        cls.moe.eval()

    def test_forward_output_shape(self):
        N = 4
        x = torch.randn(N, self.HIDDEN)
        out = self.moe(x)
        assert out.shape == (N, self.HIDDEN), \
            f"Expected ({N}, {self.HIDDEN}), got {out.shape}"

    def test_forward_no_nan(self):
        x = torch.randn(8, self.HIDDEN)
        out = self.moe(x)
        check_no_nan_inf(out, "MoELayer forward output")

    def test_router_outputs_valid(self):
        """Router should produce valid top-k weights and indices."""
        x = torch.randn(4, self.HIDDEN)
        scores = self.moe.gate(x)
        check_no_nan_inf(scores, "Router logits (scores)")

        weights, indices = self.moe._topk_routing(scores)
        check_no_nan_inf(weights, "Top-k weights")

        # Weights should be normalized (sum to 1 per token)
        weight_sums = weights.sum(dim=-1)
        expected = torch.ones_like(weight_sums)
        assert_allclose(weight_sums, expected,
                        "Top-k weights should sum to 1", atol=1e-5)

        # Indices should be in valid range
        assert (indices >= 0).all() and (indices < self.NUM_EXPERTS).all(), \
            f"Router indices out of range [0, {self.NUM_EXPERTS})"

        # Weights should be non-negative
        assert (weights >= 0).all(), "Weights should be non-negative"

    def test_forward_deterministic(self):
        x = torch.randn(4, self.HIDDEN)
        out1 = self.moe(x)
        out2 = self.moe(x)
        assert_allclose(out1, out2, "MoE forward should be deterministic")

    def test_empty_batch(self):
        x = torch.randn(0, self.HIDDEN)
        out = self.moe(x)
        assert out.shape == (0, self.HIDDEN)

    def test_single_token(self):
        x = torch.randn(1, self.HIDDEN)
        out = self.moe(x)
        assert out.shape == (1, self.HIDDEN)
        check_no_nan_inf(out, "Single-token output")

    def test_large_batch(self):
        x = torch.randn(128, self.HIDDEN)
        out = self.moe(x)
        assert out.shape == (128, self.HIDDEN)
        check_no_nan_inf(out, "Large-batch output")


# ---------------------------------------------------------------------------
# Test: MoELayer vs Reference Implementation
# ---------------------------------------------------------------------------

class TestMoEReferenceComparison:

    HIDDEN = 128
    MOE_INTERMEDIATE = 64
    NUM_EXPERTS = 4
    TOP_K = 2

    @classmethod
    def setup_class(cls):
        session_setup()
        torch.manual_seed(123)
        cls.moe = MoELayer(
            hidden_size=cls.HIDDEN,
            intermediate_size=256,
            moe_intermediate_size=cls.MOE_INTERMEDIATE,
            num_experts=cls.NUM_EXPERTS,
            top_k=cls.TOP_K,
            ep_size=1,
            ep_rank=0,
            moe_mode="ep",
            norm_topk_prob=True,
        )
        _init_moe_gate_weights(cls.moe)
        cls.moe.eval()

    def ref_forward(self, hidden_states):
        """Naive reference: iterate over (token, expert) pairs."""
        N, D = hidden_states.shape
        K = self.TOP_K

        scores = self.moe.gate(hidden_states)
        scores = F.softmax(scores, dim=-1, dtype=torch.float32)
        weights, indices = torch.topk(scores, K)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        weights = weights.to(hidden_states.dtype)

        output = torch.zeros(N, D, dtype=hidden_states.dtype)
        for i in range(N):
            for k in range(K):
                eid = indices[i, k].item()
                w = weights[i, k]
                token = hidden_states[i:i + 1]
                expert_out = self.moe.experts[eid](token).squeeze(0)
                output[i] += w * expert_out
        return output

    def test_forward_matches_reference(self):
        x = torch.randn(8, self.HIDDEN)
        moe_out = self.moe(x)
        ref_out = self.ref_forward(x)
        assert_allclose(moe_out, ref_out,
                        "MoELayer output differs from reference!")

    def test_varying_batch_sizes(self):
        for n in [1, 2, 3, 8, 16]:
            x = torch.randn(n, self.HIDDEN)
            moe_out = self.moe(x)
            ref_out = self.ref_forward(x)
            assert_allclose(moe_out, ref_out,
                            f"Batch size {n}: output differs from reference")


# ---------------------------------------------------------------------------
# Test: Split sizes logic (no dist needed)
# ---------------------------------------------------------------------------

class TestSplitSizesLogic:

    def test_bincount_correctness(self):
        ep_size = 8
        experts_per_rank = 16
        N, K = 10, 8

        torch.manual_seed(0)
        indices = torch.randint(0, 128, (N, K))
        ep_rank_target = indices // experts_per_rank

        send_counts = torch.bincount(ep_rank_target.reshape(-1), minlength=ep_size)
        assert send_counts.sum() == N * K
        assert len(send_counts) == ep_size

    def test_ep_mapping(self):
        """Expert ID → (ep_rank, local_id) mapping."""
        experts_per_rank = 32
        assert 0 // experts_per_rank == 0 and 0 % experts_per_rank == 0
        assert 63 // experts_per_rank == 1 and 63 % experts_per_rank == 31
        assert 127 // experts_per_rank == 3 and 127 % experts_per_rank == 31

    def test_unique_consecutive(self):
        ids = torch.tensor([0, 0, 0, 2, 2, 3, 3, 3, 3, 5])
        unique, counts = torch.unique_consecutive(ids, return_counts=True)
        assert unique.tolist() == [0, 2, 3, 5]
        assert counts.tolist() == [3, 2, 4, 1]

    def test_empty_bincount(self):
        ep_size = 8
        empty = torch.zeros(0, 8, dtype=torch.long)
        send_counts = torch.bincount(empty.reshape(-1), minlength=ep_size)
        assert send_counts.sum() == 0
        assert send_counts.tolist() == [0] * ep_size


# ---------------------------------------------------------------------------
# Test: Weight loader remap
# ---------------------------------------------------------------------------

class TestWeightLoaderRemap:

    def test_expert_owned_by_rank(self):
        from nanovllm.utils.loader import _remap_expert_weight
        name, eid = _remap_expert_weight(
            "model.layers.0.mlp.experts.42.gate_proj.weight",
            ep_size=8, ep_rank=2, num_experts=128,
        )
        # ep_size>1 → mapped to stacked_gate with local expert id 10
        assert name == "model.layers.0.mlp.stacked_gate", \
            f"Expected stacked_gate, got {name}"
        assert eid == 10, f"Expected local expert 10, got {eid}"

    def test_expert_not_owned(self):
        from nanovllm.utils.loader import _remap_expert_weight
        name, eid = _remap_expert_weight(
            "model.layers.0.mlp.experts.42.gate_proj.weight",
            ep_size=8, ep_rank=0, num_experts=128,
        )
        assert name is None, "Expert 42 should be skipped for ep_rank=0"
        assert eid is None

    def test_non_expert_passes_through(self):
        from nanovllm.utils.loader import _remap_expert_weight
        name, eid = _remap_expert_weight(
            "model.layers.0.self_attn.q_proj.weight",
            ep_size=8, ep_rank=0, num_experts=128,
        )
        assert name == "model.layers.0.self_attn.q_proj.weight"
        assert eid is None

    def test_ep_size_1_loads_all(self):
        from nanovllm.utils.loader import _remap_expert_weight
        for eid in [0, 42, 127]:
            name, _ = _remap_expert_weight(
                f"model.layers.0.mlp.experts.{eid}.gate_proj.weight",
                ep_size=1, ep_rank=0, num_experts=128,
            )
            assert name == f"model.layers.0.mlp.experts.{eid}.gate_proj.weight", \
                f"ep_size=1 should load expert {eid} as-is, got {name}"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_tests(instance, test_prefix="test_") -> tuple[int, int]:
    """Run all methods matching test_prefix on instance, return (passed, failed)."""
    passed, failed = 0, 0
    methods = sorted(m for m in dir(instance) if m.startswith(test_prefix))
    for method_name in methods:
        method = getattr(instance, method_name)
        try:
            method()
            print(f"  ✓ {method_name}")
            passed += 1
        except Exception as e:
            print(f"  ✗ {method_name}: {e}")
            failed += 1
    return passed, failed


# Map of all available test groups
ALL_TESTS = {
    "ExpertFFN": TestExpertFFN,
    "MoELayerEP1": TestMoELayerEP1,
    "ReferenceComparison": TestMoEReferenceComparison,
    "SplitSizes": TestSplitSizesLogic,
    "WeightLoader": TestWeightLoaderRemap,
}


if __name__ == "__main__":
    selected = [name for name in (sys.argv[1:] if len(sys.argv) > 1 else ALL_TESTS.keys())
                if name in ALL_TESTS]

    if not selected:
        print(f"Available test groups: {list(ALL_TESTS.keys())}")
        sys.exit(1)

    total_passed = 0
    total_failed = 0

    for name in selected:
        test_cls = ALL_TESTS[name]
        print(f"\n{'='*60}")
        print(f"Running: {name}")
        print(f"{'='*60}")

        # Class-level setup
        setup_method = getattr(test_cls, "setup_class", None)
        teardown_method = getattr(test_cls, "teardown_class", None)

        if setup_method:
            setup_method()

        instance = test_cls()
        p, f = run_tests(instance)
        total_passed += p
        total_failed += f

        if teardown_method:
            teardown_method()

    # Session-level cleanup
    session_teardown()

    print(f"\n{'='*60}")
    print(f"Results: {total_passed} passed, {total_failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if total_failed == 0 else 1)
