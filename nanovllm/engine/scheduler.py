from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager, HybridBlockManager
from nanovllm.layers.gdn_ops import FLA_CHUNK_SIZE


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        if config.hybrid_model:
            assert config.num_linear_states > 0, (
                "linear state slot count is not set: ModelRunner.allocate_kv_cache "
                "must run before the Scheduler is built")
            self.block_manager = HybridBlockManager(
                config.num_kvcache_blocks, config.kvcache_block_size, config.num_linear_states)
        else:
            self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.enable_decode_snapshot = config.enable_decode_snapshot
        self.decode_snapshot_interval = config.decode_snapshot_interval
        # a decode snapshot is a data dependent write, which a captured CUDA graph
        # cannot carry: only offer it in eager decode rather than silently ignore it
        assert not (self.enable_decode_snapshot and not config.enforce_eager), \
            "decode state snapshots need enforce_eager=True (CUDA graphs cannot " \
            "replay a data dependent write)"
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def plan_snapshot(self, seq: Sequence, is_decode: bool = False):
        """Ask for a state snapshot of this step, at the last complete block boundary it
        covers -- the only place where the KV cache and the linear state describe the same
        prefix. The slot is reserved here, before the step, so the ids travel with the
        sequence to every rank and the forward can write them without a copy or a message.
        """
        seq.snapshot_state, seq.snapshot_pos = -1, -1
        nct = seq.num_cached_tokens
        pos = (nct + seq.num_scheduled_tokens) // self.block_size * self.block_size
        # a decode step advances one token, so pos == nct + 1: the snapshot is the working slot
        if pos <= nct:
            return
        if is_decode:
            if not self.enable_decode_snapshot:
                return
            if pos % (self.block_size * self.decode_snapshot_interval):
                return
        elif (pos - nct) % FLA_CHUNK_SIZE:
            # the chunked scan only knows per-chunk boundaries; a safety net, since
            # num_cached_tokens is kept block aligned (decode needs no such alignment)
            return
        if self.block_manager.alloc_snapshot(seq) >= 0:
            seq.snapshot_pos = pos

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
        num_batched_tokens = 0

        # prefill
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                break
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            if seq.num_cached_tokens + seq.num_scheduled_tokens < seq.num_tokens:
                # a later chunk can only snapshot where KV and state cover the same prefix,
                # so keep chunk ends on a block boundary (usually a no-op)
                end = (seq.num_cached_tokens + seq.num_scheduled_tokens) \
                    // self.block_size * self.block_size
                if end > seq.num_cached_tokens:
                    seq.num_scheduled_tokens = end - seq.num_cached_tokens
            self.plan_snapshot(seq)
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                self.plan_snapshot(seq, is_decode=True)
                scheduled_seqs.append(seq)
        assert scheduled_seqs or self.is_finished()
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            # register the snapshot the forward just wrote, now that num_cached_tokens and
            # the block hashes are up to date
            self.block_manager.register_snapshot(seq)
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                # chunked prefill: the next chunk re-reads what this one wrote
                self.block_manager.carry_state(seq)
                continue
            elif is_prefill:
                self.block_manager.release_read_slot(seq)
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
