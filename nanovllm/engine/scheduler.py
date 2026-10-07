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
        self.async_scheduling = config.async_scheduling
        # async scheduling: per-request device slots holding the latest sampled token
        # (ModelRunner.token_of_slot); handed out on the first prefill, returned when
        # the sequence finishes
        self.slot_free = deque(range(config.max_num_seqs))
        # sequences handed to a batch but not yet committed by postprocess: they sit in no
        # queue (the batch owns them), so a round that schedules nothing has not
        # necessarily finished -- see is_finished
        self.num_inflight = 0

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
        return not self.waiting and not self.running and not self.num_inflight

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
        num_batched_tokens = 0

        # prefill
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            if self.async_scheduling and seq.slot_id < 0 and not self.slot_free:
                # every scheduled sequence owns a device slot, and that pool is global
                # while this loop's cap is per batch: sequences beyond it wait here (the
                # queue's whole purpose) until a finishing sequence hands its slot back
                break
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
            if self.async_scheduling and seq.slot_id < 0:
                seq.slot_id = self.slot_free.popleft()
            # the batch owns this sequence until postprocess commits it: a completed one
            # enters `running` there and a mid-chunk one goes back to `waiting` -- that is
            # what lets a prefill batch stay in flight for a step
            self.waiting.popleft()
            scheduled_seqs.append(seq)
            if seq.num_cached_tokens + seq.num_scheduled_tokens < seq.num_tokens:
                # chunked prefill: the loop used to re-read this same seq here and stop on
                # the exhausted token budget -- keep the batch composition identical
                break

        if scheduled_seqs:
            self.num_inflight += len(scheduled_seqs)
            return scheduled_seqs, True

        # decode
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            # async scheduling: this step samples the token that becomes the sequence's
            # next input. If a sampling event is still in flight, that token is not in
            # token_ids yet -- reserve its slot now (optimistically advance the length)
            # so that every length / block formula below keeps working unchanged.
            # postprocess fills the reservation one step later.
            placeholder = self.async_scheduling and seq.pending > 0
            if placeholder:
                seq.token_ids.append(-1)
                seq.num_tokens += 1
                seq.placeholder = True
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    if placeholder:     # roll the reservation back before it leaves
                        seq.token_ids.pop()
                        seq.num_tokens -= 1
                        seq.placeholder = False
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                if self.async_scheduling:
                    seq.pending += 1
                self.block_manager.may_append(seq)
                self.plan_snapshot(seq, is_decode=True)
                scheduled_seqs.append(seq)
        assert scheduled_seqs or self.is_finished() or self.num_inflight, \
            "nothing scheduled and nothing in flight: the scheduler is stuck"
        self.num_inflight += len(scheduled_seqs)
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        # async scheduling: reservations only live inside one schedule() call, so a
        # sequence entering the waiting queue never carries an unfilled placeholder;
        # its still-in-flight token is dropped by postprocess (status != RUNNING)
        assert not seq.placeholder
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        self.num_inflight -= len(seqs)
        for seq, token_id in zip(seqs, token_ids):
            if not is_prefill and seq.status is not SequenceStatus.RUNNING:
                # async scheduling: the sequence finished or was preempted while this
                # step was already in flight; its sampled token is stale, drop it.
                # (a mid-chunk prefill sequence is legitimately still WAITING here)
                if seq.pending > 0:
                    seq.pending -= 1
                seq.num_scheduled_tokens = 0
                continue
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            # register the snapshot the forward just wrote, now that num_cached_tokens and
            # the block hashes are up to date
            self.block_manager.register_snapshot(seq)
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                # chunked prefill: the next chunk re-reads what this one wrote; the batch
                # is done with the sequence, so it goes back to the queue it came from
                self.block_manager.carry_state(seq)
                self.waiting.appendleft(seq)
                continue
            elif is_prefill:
                self.block_manager.release_read_slot(seq)
                # prefill committed: the sequence joins the running set here (the in-flight
                # batch owned it until now), before the finish check below can remove it
                seq.status = SequenceStatus.RUNNING
                self.running.append(seq)
            if seq.placeholder:
                # async scheduling: token_ids[-1] is the reservation taken at schedule
                # time for this in-flight token -- fill it in place, do not append
                seq.token_ids[-1] = token_id
                seq.last_token = token_id
                seq.placeholder = False
            else:
                seq.append_token(token_id)
            if seq.pending > 0:
                seq.pending -= 1
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
                if self.async_scheduling and seq.slot_id >= 0:
                    # the slot can go back right away: a still-in-flight scatter into it
                    # is ordered before any new owner's write on the same stream
                    self.slot_free.append(seq.slot_id)
                    seq.slot_id = -1
