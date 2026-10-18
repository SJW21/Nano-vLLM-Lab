"""Shrink-only bookkeeping for DP result reporting.

The run-ahead pipeline can consume a batch that was scheduled *before* a sequence
finished: `postprocess` then sees the sequence already FINISHED and would report it a
second time. The only batch that can still repeat a freshly reported id is the one in
flight at the moment of the report -- every batch scheduled afterwards already saw the
sequence leave `running`. So the set does not grow forever: each reported id is bucketed
under that in-flight batch's handle and the whole bucket is dropped when the batch is
consumed.

Zero dependencies on purpose: the local CPU test loads this file by path.
"""


class ReportedTracker:
    """seq_ids already reported, tagged with the in-flight batch that may repeat them."""

    def __init__(self):
        self._buckets: dict[object, set[int]] = {}

    def filter_and_report(self, seqs, inflight_handle) -> list:
        """Return the finished sequences that have not been reported yet.

        `inflight_handle` names the batch that may still contain them; `None` means
        nothing is in flight, so there is no batch to tag and ids are reported directly.
        """
        pending = self._buckets.get(inflight_handle) if inflight_handle is not None else None
        finished = [seq for seq in seqs
                    if seq.is_finished and (pending is None or seq.seq_id not in pending)]
        if finished and inflight_handle is not None:
            if pending is None:
                pending = self._buckets[inflight_handle] = set()
            pending.update(seq.seq_id for seq in finished)
        return finished

    def on_batch_consumed(self, handle):
        """A batch was consumed; it can no longer repeat ids, so its bucket goes away."""
        self._buckets.pop(handle, None)

    def is_empty(self) -> bool:
        return not self._buckets
