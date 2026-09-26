import torch
from dataclasses import dataclass


@dataclass(slots=True)
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    state_ids: torch.Tensor | None = None
    cached_states: torch.Tensor | None = None
    # this step's linear-state snapshot, one entry per sequence: the destination slot
    # (-1 => none) and, for prefill, the flat-batch token index it takes state from.
    # ``has_snapshot`` is the host gate -- probing the tensors would sync per layer.
    snapshot_ids: torch.Tensor | None = None
    snapshot_pos: torch.Tensor | None = None
    has_snapshot: bool = False
    # chunked-scan metadata for the GDN layers, filled on the first one of a prefill step
    chunk_indices: torch.Tensor | None = None
    chunk_offsets: torch.Tensor | None = None

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None,
                state_ids=None, cached_states=None, snapshot_ids=None, snapshot_pos=None, has_snapshot=False):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables,
                       state_ids, cached_states, snapshot_ids, snapshot_pos, has_snapshot)

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
