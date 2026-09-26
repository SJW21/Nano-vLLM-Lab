from collections import deque
import xxhash
import numpy as np
import torch

from nanovllm.engine.sequence import Sequence
from nanovllm.config import Config


class Block:
    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class LinearState:
    def __init__(self, state_id):
        self.state_id = state_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []   


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self) -> int:
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    def _deallocate_block(self, block_id: int):
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> int:
        h = -1
        num_cached_blocks = 0
        num_new_blocks = seq.num_blocks
        for i in range(seq.num_blocks - 1):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break
            num_cached_blocks += 1
            if block_id in self.used_block_ids:
                num_new_blocks -= 1
        if len(self.free_block_ids) < num_new_blocks:
            return -1
        return num_cached_blocks

    def allocate(self, seq: Sequence, num_cached_blocks: int):
        assert not seq.block_table
        h = -1
        for i in range(num_cached_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id[h]
            block = self.blocks[block_id]
            if block_id in self.used_block_ids:
                block.ref_count += 1
            else:
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)
        for i in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(self._allocate_block())
        seq.num_cached_tokens = num_cached_blocks * self.block_size

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self._allocate_block())

    def hash_blocks(self, seq: Sequence):
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        if start == end: return
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block.block_id

    def carry_state(self, seq: Sequence):
        """Non-hybrid counterpart of ``HybridBlockManager.carry_state``."""
        return None

    def alloc_snapshot(self, seq: Sequence) -> int:
        """No linear state, hence nothing to snapshot. Must be an int, not ``None``:
        the scheduler compares the result against 0."""
        return -1

    def register_snapshot(self, seq: Sequence):
        """Non-hybrid counterpart of ``HybridBlockManager.register_snapshot``."""
        return None

    def release_read_slot(self, seq: Sequence):
        """Non-hybrid counterpart of ``HybridBlockManager.release_read_slot``."""
        return None


class HybridBlockManager(BlockManager):
    # hybrid Attention Manager: Linear (mamba state) + Full

    def __init__(self, num_blocks, block_size, num_states):
        super().__init__(num_blocks, block_size)
        # state ids index MambaPool rows directly, so this is that pool's slot count
        assert num_states > 0, "linear state pool is empty; did allocate_kv_cache run?"
        self.states: list[LinearState] = [LinearState(i) for i in range(num_states)]
        self.free_state_ids: deque[int] = deque(range(num_states))
        self.used_state_ids: set[int] = set()
        self.hash_table = {
            "full": self.hash_to_block_id,
            "linear": dict()
        }

    def _allocate_state(self) -> int:
        state_id = self.free_state_ids.popleft()
        state = self.states[state_id]
        assert state.ref_count == 0
        if state.hash != -1 and self.hash_table['linear'].get(state.hash) == state_id:
            del self.hash_table['linear'][state.hash]
        state.reset()
        self.used_state_ids.add(state_id)
        return state_id

    def _deallocate_state(self, state_id: int):
        assert self.states[state_id].ref_count == 0
        self.used_state_ids.remove(state_id)
        self.free_state_ids.append(state_id)

    def can_allocate(self, seq: Sequence) -> int:
        # Judge block/state is enough & find the cached blocks(state)
        h = -1
        num_cached_blocks = 0
        num_new_blocks = seq.num_blocks
        final_cached_blocks = 0
        final_new_blocks = 0
        # step 1: cached full attn (full is always the first)
        for i in range(seq.num_blocks - 1):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_table['full'].get(h, -1)
            state_id = self.hash_table['linear'].get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break
            num_cached_blocks += 1
            if block_id in self.used_block_ids:
                num_new_blocks -= 1
            if state_id != -1 and self.states[state_id].token_ids == token_ids:
                final_cached_blocks = num_cached_blocks
                final_new_blocks = num_new_blocks
            
        if len(self.free_block_ids) < final_new_blocks or \
            len(self.free_state_ids) < 1:
            return -1
        return final_cached_blocks

    def allocate(self, seq: Sequence, num_cached_blocks: int):
        assert not seq.block_table
        assert seq.cached_state == -1 and seq.mamba_state == -1

        def revive_and_count_increase(id: int, is_full: bool):           
            data = self.blocks[id] if is_full else self.states[id]
            used_list = self.used_block_ids \
                if is_full else self.used_state_ids
            if id in used_list:
                data.ref_count += 1
            else:
                # reviving an entry that was released but never overwritten
                data.ref_count = 1
                if is_full:
                    self.free_block_ids.remove(id)
                    self.used_block_ids.add(id)
                else:
                    self.free_state_ids.remove(id)
                    self.used_state_ids.add(id)

        h = -1
        for i in range(num_cached_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            # full
            block_id = self.hash_table['full'][h]
            revive_and_count_increase(block_id, True)
            seq.block_table.append(block_id)
            # linear: only the *frontier* block's state is revived -- it covers every token
            # before it, and the earlier entries are read by nobody (reviving leaks refs)
            if i == num_cached_blocks - 1:
                state_id = self.hash_table['linear'].get(h, -1)
                if state_id != -1 and self.states[state_id].token_ids == token_ids:
                    revive_and_count_increase(state_id, False)
                    seq.cached_state = state_id

        for i in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(self._allocate_block())
        seq.num_cached_tokens = num_cached_blocks * self.block_size
        # new linear state buffer
        seq.mamba_state = self._allocate_state()

    def reckoning(self, id, is_full: bool):
        data = self.blocks[id] if is_full else self.states[id]
        data.ref_count -= 1
        if data.ref_count == 0:
            if is_full:
                self._deallocate_block(id)
            else:
                self._deallocate_state(id)

    def release_read_slot(self, seq: Sequence):
        """Drop the read-in slot of a sequence (idempotent). ``cached_state`` may alias
        ``mamba_state`` mid-chunked-prefill -- that slot is the working state, keep it."""
        if seq.cached_state != -1 and seq.cached_state != seq.mamba_state:
            self.reckoning(seq.cached_state, False)
        seq.cached_state = -1

    def carry_state(self, seq: Sequence):
        """Chunked prefill continuation: the next chunk re-reads what this one wrote, so
        the working slot becomes both source and target."""
        self.release_read_slot(seq)
        seq.cached_state = seq.mamba_state

    def alloc_snapshot(self, seq: Sequence):
        """Reserve the slot this step's forward will write a snapshot into, *before* the
        step, so every rank learns the destination from the sequence itself. It stays held
        (ref_count 1) until ``register_snapshot`` hands it to the hash table."""
        seq.snapshot_state = self._allocate_state() if self.free_state_ids else -1
        return seq.snapshot_state

    def register_snapshot(self, seq: Sequence):
        """Turn the slot the forward just wrote into a prefix-cache entry: must run after
        ``hash_blocks`` (the key is the block the snapshot sits on) and after
        ``num_cached_tokens`` advanced. The slot is released right away -- as with a block,
        the hash entry keeps it revivable until some other token takes it."""
        slot, pos = seq.snapshot_state, seq.snapshot_pos
        seq.snapshot_state, seq.snapshot_pos = -1, -1
        if slot < 0:
            return
        # state and KV must cover the same prefix: only a *complete* block boundary counts
        if pos <= 0 or pos > seq.num_cached_tokens or pos % self.block_size:
            self.reckoning(slot, False)
            return
        block = self.blocks[seq.block_table[pos // self.block_size - 1]]
        state = self.states[slot]
        state.update(block.hash, block.token_ids)
        self.hash_table['linear'][block.hash] = slot
        self.reckoning(slot, False)

    def deallocate(self, seq: Sequence):
        # when req finished or preempted
        for block_id in reversed(seq.block_table):
            self.reckoning(block_id, True)
        seq.num_cached_tokens = 0
        seq.block_table.clear()
        # linear state
        if seq.snapshot_state != -1:
            # reserved but never registered (preempted / aborted mid-step)
            self.reckoning(seq.snapshot_state, False)
            seq.snapshot_state = -1
        self.release_read_slot(seq)
        self.reckoning(seq.mamba_state, False)
        seq.refresh_linear_state()


class MambaShape:
    def __init__(self, config: Config):
        hf_conf = config.text_config   # linear_* live on the text half (see Config)
        tp_size = config.tensor_parallel_size
        self.nhead_k = hf_conf.linear_num_key_heads
        self.nhead_v = hf_conf.linear_num_value_heads
        self.k_dim = hf_conf.linear_key_head_dim
        self.v_dim = hf_conf.linear_value_head_dim
        self.conv_kernel_size = hf_conf.linear_conv_kernel_dim

        # shape check
        assert self.nhead_k % tp_size == 0, f"linear num key heads {self.nhead_k} \
            cannot be sharded by tp_size {tp_size}"
        assert self.nhead_v % tp_size == 0, f"linear num query heads {self.nhead_v} \
            cannot be sharded by tp_size {tp_size}"
        k_dim_all = self.nhead_k * self.k_dim
        v_dim_all = self.nhead_v * self.v_dim

        # conv state
        conv_dim = v_dim_all + 2 * k_dim_all
        self.conv_state_shape = conv_dim // tp_size, self.conv_kernel_size - 1
        # temporal state
        self.temporal_state_shape = self.nhead_v // tp_size, self.v_dim, self.k_dim

        # dtype setting
        dtype_map = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }
        dtype_size = {
            torch.float32: 4,
            torch.bfloat16: 2,
            torch.float16: 2,
        }
        self.conv_dtype = torch.bfloat16
        # Qwen3.5 has no mamba_ssm_dtype (a Mamba-family field), and the GDN kernels are
        # written for an fp32 state throughout
        self.ssm_dtype = dtype_map[getattr(hf_conf, "mamba_ssm_dtype", "float32")]

        self.conv_bytes = self.conv_state_shape[0] * self.conv_state_shape[1] * dtype_size[self.conv_dtype]
        self.tempotal_bytes = self.temporal_state_shape[0] * self.v_dim * self.k_dim * dtype_size[self.ssm_dtype]


class MambaPool:
    def __init__(self,
                 config: Config,
                 mem_size: int,
                 num_layers: int,
                 device: torch.device
                 ):
        shape_param = MambaShape(config)
        state_bytes = num_layers * (shape_param.conv_bytes + shape_param.tempotal_bytes)
        size = mem_size // state_bytes

        # a slot is one row of both tensors (over all layers), and that row index *is* the
        # id the kernels see: padding uses the -1 sentinel, which every kernel guards, so
        # no slot is kept in reserve
        assert size > 0, f"linear cache holds no state: {mem_size} < {state_bytes}"
        self.num_slots = size

        self.conv_state = torch.zeros(
                        size=(num_layers, size) + shape_param.conv_state_shape,
                        dtype=shape_param.conv_dtype,
                        device=device,
                    )

        self.temporal_state = torch.zeros(
                    size=(num_layers, size) + shape_param.temporal_state_shape,
                    dtype=shape_param.ssm_dtype,
                    device=device,
                )