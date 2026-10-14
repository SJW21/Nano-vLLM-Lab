from copy import copy
from enum import Enum, auto
from itertools import count
from dataclasses import dataclass

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class SequenceManager:
    # sequence id counter
    def __init__(self):
        self.counter = count()
    
    @property
    def get_id(self):
        return next(self.counter)


@dataclass
class TokenizedInput:
    # token_ids
    token_ids: list[int]
    # id of sequence
    seq_id: int
    # Params for Sampler
    sampling_params: SamplingParams


class Sequence:
    block_size = 256
    counter = count()

    def __init__(self, token_ids: list[int], seq_id: int, sampling_params = SamplingParams()):
        self.seq_id = seq_id
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.num_scheduled_tokens = 0
        self.is_prefill = True
        self.block_table = []
        # slot the next prefill reads its state from: a prefix-cache slot, or this
        # sequence's own mamba_state while a chunked prefill is in flight (-1 => zero)
        self.cached_state = -1
        self.mamba_state = -1 # this sequence's working linear state slot
        # this step's snapshot target and position, picked by the scheduler before the
        # step so every rank writes the same slot (both -1 => no snapshot)
        self.snapshot_state = -1
        self.snapshot_pos = -1
        # async scheduling bookkeeping (see Scheduler.schedule / postprocess): slot_id
        # is this sequence's row in ModelRunner.token_of_slot, `pending` counts in-flight
        # decode steps, `placeholder` marks a reserved (unfilled) token_ids[-1]
        self.slot_id = -1
        self.pending = 0
        self.placeholder = False
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def refresh_linear_state(self):
        self.cached_state = -1
        self.mamba_state = -1
        self.snapshot_state = -1
        self.snapshot_pos = -1

    def __getstate__(self):
        # decode ships only last_token (a worker never reads the full list back) and
        # slot_id travels, so every rank scatters into the same token_of_slot row
        last_state = self.last_token if not self.is_prefill else self.token_ids
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, self.cached_state, self.mamba_state, self.snapshot_state, self.snapshot_pos, last_state, self.slot_id)

    def __setstate__(self, state):
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, self.cached_state, self.mamba_state, self.snapshot_state, self.snapshot_pos, last_state, self.slot_id = state
        if isinstance(last_state, list):
            self.token_ids = last_state
            self.last_token = self.token_ids[-1]
        else:
            self.token_ids = []
            self.last_token = last_state
