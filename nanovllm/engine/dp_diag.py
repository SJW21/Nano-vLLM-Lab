"""Env-gated diagnostic channel for DP differential runs (NANOVLLM_DP_DIAG).

The DP clients live in spawn'ed processes, so test-side monkeypatching cannot reach
their Scheduler / ModelRunner. install() hooks those classes inside the client process
and the diag handle writes <dir>/client_<rank>.json at exit: the schedule / prefill /
handed / input-token records the e2e parent compares, plus the end-of-run invariants.
NANOVLLM_DP_DIAG_MODE=invariants keeps only the invariants -- no hooks and empty
records, so a benchmark run does not pay for per-row tracing; unset means the full set.
NANOVLLM_DETERMINISTIC_SAMPLING=1 replaces sampling with argmax, proven on tied logits
before any comparison, independent of the mode. With the env unset install() returns
None and nothing runs.
"""
import json
import os

import torch

from nanovllm.engine.sequence import SequenceStatus

_STATE = None


def install(client):
    """Hook this client process; returns the diag handle, or None when the channel is off."""
    global _STATE
    diag_dir = os.environ.get("NANOVLLM_DP_DIAG")
    if not diag_dir:
        return None
    mode = os.environ.get("NANOVLLM_DP_DIAG_MODE", "full")
    assert mode in ("full", "invariants"), \
        f"unknown NANOVLLM_DP_DIAG_MODE={mode!r} (expected 'full' or 'invariants')"
    assert _STATE is None, "dp_diag is installed once per client process"
    _STATE = {
        "dir": diag_dir,
        "rank": client.rank,
        "client": client,
        "mode": "async" if client.async_scheduling else "sync",
        "schedule": [],
        "prefill": [],
        "preempt": 0,
        "handed": {},
        "trace": [],
        "prompt_len": {},
        "argmax_probe": False,
    }
    if mode == "full":
        _install_scheduler_hooks()
        _install_runner_hooks()
    if os.environ.get("NANOVLLM_DETERMINISTIC_SAMPLING") == "1":
        _STATE["argmax_probe"] = _install_argmax(client.model_runner)
    return _Diag()


class _Diag:

    def finalize(self):
        global _STATE
        state, _STATE = _STATE, None
        if state is None:
            return
        for rows in state["trace"]:
            for row in rows:
                row[2] = int(row[2])
        client = state["client"]
        data = {
            "mode": state["mode"],
            "schedule": state["schedule"],
            "prefill": state["prefill"],
            "preempt": state["preempt"],
            "handed": state["handed"],
            "trace": state["trace"],
            "prompt_len": state["prompt_len"],
            "invariants": _invariants(client),
            "peak_alloc_mib": torch.cuda.max_memory_allocated(device=client.rank) / 2 ** 20,
            "argmax_probe": state["argmax_probe"],
        }
        path = os.path.join(state["dir"], f"client_{state['rank']}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)


def _install_scheduler_hooks():
    from nanovllm.engine.scheduler import Scheduler

    orig_schedule = Scheduler.schedule

    def schedule(self):
        seqs, is_prefill = orig_schedule(self)
        _STATE["schedule"].append([len(seqs), is_prefill, [seq.seq_id for seq in seqs], None,
                                   [seq.num_tokens for seq in seqs]])
        return seqs, is_prefill

    orig_postprocess = Scheduler.postprocess

    def postprocess(self, seqs, token_ids, is_prefill):
        if seqs:
            # the batch being consumed is the oldest non-empty one not yet marked
            # (the pipeline is one deep); its discards are listed per batch
            record = next((entry for entry in _STATE["schedule"]
                           if entry[0] and entry[3] is None), None)
            assert record is not None, "postprocess without a matching scheduled batch"
            record[3] = [seq.seq_id for seq in seqs
                         if not is_prefill and seq.status is not SequenceStatus.RUNNING]
            for seq, token_id in zip(seqs, token_ids):
                _STATE["handed"].setdefault(seq.seq_id, []).append(int(token_id))
        return orig_postprocess(self, seqs, token_ids, is_prefill)

    orig_preempt = Scheduler.preempt

    def preempt(self, seq):
        _STATE["preempt"] += 1
        return orig_preempt(self, seq)

    Scheduler.schedule = schedule
    Scheduler.postprocess = postprocess
    Scheduler.preempt = preempt


def _install_runner_hooks():
    from nanovllm.engine.model_runner import ModelRunner

    orig_prefill = ModelRunner.prepare_prefill

    def prepare_prefill(self, seqs):
        for seq in seqs:
            # seq_id first: a re-prefill after preemption is a second entry for the
            # same sequence, and the parent needs the prompt length for every one
            _STATE["prefill"].append([seq.seq_id, seq.num_cached_tokens,
                                      seq.num_scheduled_tokens, seq.cached_state])
            _STATE["prompt_len"][seq.seq_id] = seq.num_prompt_tokens
        return orig_prefill(self, seqs)

    orig_launch_decode = ModelRunner.launch_decode

    def launch_decode(self, seqs):
        _STATE["trace"].append([_trace_row(seq, self.token_of_slot[seq.slot_id].clone())
                                for seq in seqs])
        return orig_launch_decode(self, seqs)

    orig_prepare_decode = ModelRunner.prepare_decode

    def prepare_decode(self, seqs):
        _STATE["trace"].append([_trace_row(seq, seq.last_token) for seq in seqs])
        return orig_prepare_decode(self, seqs)

    ModelRunner.prepare_prefill = prepare_prefill
    ModelRunner.launch_decode = launch_decode
    ModelRunner.prepare_decode = prepare_decode


def _trace_row(seq, token):
    # the input token this row will read, and where it sits in the handed stream
    record = _STATE["handed"].setdefault(seq.seq_id, [])
    return [seq.seq_id, seq.num_tokens, token, len(record), None, None, None]


def _install_argmax(runner):
    from nanovllm.layers.sampler import Sampler

    def forward(self, logits, temperatures):
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        return logits.argmax(dim=-1)

    Sampler.forward = forward
    probe_logits = torch.zeros(16, 2)          # top-2 exactly tied
    probe_temps = torch.full((16,), 1e-6)
    sampler = runner.sampler
    got = sampler(probe_logits, probe_temps)
    again = sampler(probe_logits, probe_temps)
    if got.tolist() != [0] * 16 or not got.equal(again):
        raise SystemExit(
            "[dp-diag] deterministic sampling is NOT in effect: the argmax probe returned "
            f"{got.tolist()[:4]}... on tied logits -- refusing to compare tokens")
    return True


def _invariants(client):
    sched = client.scheduler
    running = list(sched.running)
    return {
        "num_inflight": sched.num_inflight,
        "waiting": len(sched.waiting),
        "running": len(running),
        "slot_free": len(sched.slot_free),
        "max_num_seqs": sched.max_num_seqs,
        "pending_total": sum(seq.pending for seq in running),
        "placeholder_left": any(seq.placeholder for seq in running),
        "staged_pending": client._pending is not None,
        "staged_inflight": client._inflight is not None,
    }
