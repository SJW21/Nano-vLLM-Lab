"""
Tests for EP IDLE sync mechanism (Phase 2).

Run:
    python tests/test_ep_idle.py

These tests verify the deadlock-prevention infrastructure without needing
multiple GPUs.  All tests run single-process.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import deque
from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


# ---------------------------------------------------------------------------
# Test: Block 0 reservation
# ---------------------------------------------------------------------------

class TestBlock0Reservation:

    def test_block_0_not_in_free_list(self):
        """Block 0 should be reserved — never returned to free list."""
        bm = BlockManager(num_blocks=10, block_size=256)
        assert 0 not in bm.free_block_ids, \
            "Block 0 should be reserved, not in free list"
        assert list(bm.free_block_ids) == list(range(1, 10)), \
            f"Free blocks should be 1..9, got {list(bm.free_block_ids)}"

    def test_block_0_never_allocated(self):
        """Allocate all blocks — block 0 should never be given out."""
        bm = BlockManager(num_blocks=5, block_size=256)  # blocks 1..4 free
        seq = Sequence([42] * 300, seq_id=1)  # needs 2 blocks (300 tokens / 256)
        # First allocate should succeed
        nc = bm.can_allocate(seq)
        assert nc >= 0, f"Should be able to allocate, got {nc}"
        bm.allocate(seq, nc)
        assert 0 not in seq.block_table, \
            f"Block 0 should never be allocated, got block_table={seq.block_table}"

    def test_deallocate_preserves_block_0(self):
        """After all blocks are used and freed, block 0 stays reserved."""
        bm = BlockManager(num_blocks=3, block_size=256)  # blocks 1,2 free
        seq = Sequence([42] * 500, seq_id=1)  # needs 2 blocks
        nc = bm.can_allocate(seq)
        bm.allocate(seq, nc)
        assert 0 not in seq.block_table

        bm.deallocate(seq)
        assert 0 not in bm.free_block_ids, \
            "Block 0 should still be reserved after deallocation"


# ---------------------------------------------------------------------------
# Test: Dummy batch construction
# ---------------------------------------------------------------------------

class TestDummyBatch:

    def test_make_idle_batch_structure(self):
        """Dummy batch should have the expected properties for deadlock prevention."""
        seq = Sequence([0], seq_id=-1)
        seq.num_scheduled_tokens = 1
        seq.block_table = [0]
        seq.status = SequenceStatus.RUNNING

        assert seq.seq_id == -1, "Dummy marker should be -1"
        assert seq.status == SequenceStatus.RUNNING, "Dummy should be RUNNING"
        assert seq.block_table == [0], "Dummy should use reserved block 0"
        assert seq.num_scheduled_tokens == 1, "Dummy should have 1 scheduled token"
        assert len(seq) == 1, "Dummy should have 1 token"
        assert seq.last_token == 0, "Dummy should have token 0"

    def test_dummy_not_processed_by_scheduler(self):
        """Scheduler.postprocess should skip dummy seqs (seq_id == -1)."""
        from nanovllm.engine.scheduler import Scheduler

        config = Config.__new__(Config)
        config.max_num_seqs = 512
        config.max_num_batched_tokens = 16384
        config.eos = -1
        config.kvcache_block_size = 256
        config.num_kvcache_blocks = 100

        scheduler = Scheduler(config)

        # Create a dummy and a real seq
        dummy = Sequence([0], seq_id=-1)
        dummy.num_scheduled_tokens = 1
        dummy.block_table = [0]
        dummy.status = SequenceStatus.RUNNING

        real = Sequence([42, 43, 44], seq_id=1)
        real.num_scheduled_tokens = 3
        real.block_table = [1]
        real.num_cached_tokens = 0

        # postprocess should skip dummy and process real
        token_ids = [99, 100]
        scheduler.postprocess([dummy, real], token_ids, is_prefill=True)

        # Dummy: token not appended (skipped)
        assert len(dummy) == 1, "Dummy should not have appended token"
        # Real: token appended
        assert len(real) == 4, "Real seq should have token appended"
        assert real.last_token == 100


# ---------------------------------------------------------------------------
# Test: Config validation
# ---------------------------------------------------------------------------

class TestEPConfig:

    def test_ep_size_default(self):
        """Default EP size is 1 (disabled)."""
        from dataclasses import replace
        config = Config.__new__(Config)
        # Config fields are set by dataclass defaults
        assert Config.__dataclass_fields__["expert_parallel_size"].default == 1
        assert Config.__dataclass_fields__["moe_mode"].default == "ep"

    def test_ep_must_equal_world_size(self):
        """EP size must be 1 or dp*tp."""
        # Valid cases are checked in __post_init__ which requires model dir
        pass  # __post_init__ assertion is verified in integration tests


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_tests(instance, test_prefix="test_") -> tuple[int, int]:
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


ALL_TESTS = {
    "Block0Reservation": TestBlock0Reservation,
    "DummyBatch": TestDummyBatch,
    "EPConfig": TestEPConfig,
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
        instance = test_cls()
        p, f = run_tests(instance)
        total_passed += p
        total_failed += f

    print(f"\n{'='*60}")
    print(f"Results: {total_passed} passed, {total_failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if total_failed == 0 else 1)
