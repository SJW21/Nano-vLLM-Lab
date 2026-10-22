"""4-GPU MoE (TP/EP) end-to-end logits gate.

Runs three stages, each in its own process so GPU memory is fully released
between them:

    driver --(prompt.pt)--> nano --(nano_<mode>.pt)--> ref --(ref_<mode>.pt)--> compare

nano:    LLMEngine with tensor_parallel_size=TP, expert_parallel_size=EP.
         Captures every logits tensor the runner produces for a fixed prompt:
         the prefill step (ParallelLMHead returns only the LAST position of
         each sequence) and each decode step (one position each).
ref:     transformers, bf16, teacher-forced on the concatenated token sequence
         (prompt + the tokens nano actually fed back at decode), selecting the
         same positions.
compare: top-1 / top-5 / mean|diff| / max|diff| / cosine / rel-L2.

Why logits and not generated tokens: the sampler is Gumbel-max with no seed
and temperature=0 is asserted away, so token equality is not a reliable gate.

Usage (server, from the repo root; --model is required):

    python testfile/test_moe_tp_gate.py \
        --model <model-dir> \
        --modes eager,graph

    # one mode only, more decode steps, longer prompt
    python testfile/test_moe_tp_gate.py --model ... --modes graph --max-tokens 8 --prompt-len 128

Exit code 0 iff every requested mode passes. A JSON report is written to
<out-dir>/report_<mode>.json.
"""

import argparse
import json
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# A paragraph of ordinary text. Both stages consume the SAME token ids (the
# driver tokenizes once), so content only matters for having confident
# predictions rather than near-ties.
DEFAULT_PROMPT = (
    "The history of computing is a history of abstraction. Each generation of "
    "machines hid more of the previous generation's details behind a simpler "
    "interface, and each layer of abstraction made the next layer possible. "
    "Compilers hid registers, operating systems hid disks, and virtual machines "
    "hid whole computers. The same pattern appears in language models: kernels "
    "hide memory movement, runtimes hide scheduling, and serving engines hide "
    "the distributed hardware underneath. What remains visible to the programmer "
    "is the model itself, and what remains visible to the model is a sequence of "
    "tokens. Everything else is an implementation detail, until it is not: an "
    "implementation detail becomes a correctness bug the moment its assumptions "
    "stop holding, which is why the boundary between layers is where most of the "
    "difficulty lives."
)


# ---------------------------------------------------------------------------
# Criterion notes (kept next to the code so a future threshold change has to
# confront the reasoning, and mirrored into every JSON report)
# ---------------------------------------------------------------------------
#
# Both sides run bf16, so this comparison is precision-symmetric.  The residual
# difference comes from different kernels (FlashAttention vs SDPA/eager), TP
# all-reduce summation order, and incremental decode vs a full-sequence
# forward.  Those are ~1e-3 relative L2 in practice.
#
# rel_l2 (primary numerical gate vs the transformers reference):
#   CALIBRATED FROM A REAL RUN (Qwen3-30B-A3B, TP=4/EP=4, 64-token prompt,
#   4 steps): nano eager and nano graph logits are BIT-IDENTICAL, and both
#   differ from the transformers bf16 reference by rel_l2 1.1e-2 .. 2.6e-2
#   (mean 2.0e-2) with top1 4/4 and cosine >= 0.9996.  That spread comes from
#   48 layers of bf16 plus EP all-reduce / kernel differences, not from a
#   defect.  5e-2 leaves ~1.9x headroom.  The earlier 1e-2 default was an
#   estimate (~1e-3) and is superseded by this measurement.
#   Graph correctness is gated separately and much tighter: nano graph vs
#   nano eager must agree to --graph-rel-l2-max (default 1e-5; observed 0).
#   Gated on the WORST step, not the mean, because averaging would dilute a
#   single stale replay (e.g. one graph reading capture-time addresses).
#   mean_abs is kept but NOT promoted: it is scale-dependent (logits live in
#   roughly +-10..30), so the long-standing 1.0 threshold was close to vacuous.
#   rel_l2 supplies the scale-invariant bound that mean_abs cannot.
#
# top1 (kept, discrete):
#   argmax is noise-sensitive: when the reference's top-2 margin is within
#   bf16 error, a flip is an artifact rather than a bug.  The 0.75 threshold is
#   unchanged, and n = max_tokens + 1 (default 5 steps), i.e. it tolerates one
#   mismatch.  Every row now reports top1_margin so a failure can be
#   classified as a near-tie artifact vs. a real regression instead of
#   silently re-tuning this number.
#
# cosine (kept), mean_abs (kept), top5_overlap (diagnostic only):
#   Unchanged.  top5_overlap is printed and reported but is not a gate; it is
#   useful context when investigating a top1 mismatch.
CRITERION_NOTES = (
    "rel_l2 (worst step, 5e-2) is calibrated from a real run: Qwen3-30B-A3B, "
    "TP=4/EP=4, 64-token prompt, 4 steps. nano eager and nano graph logits were "
    "bit-identical; both differed from the transformers bf16 reference by "
    "rel_l2 1.1e-2..2.6e-2 (mean 2.0e-2, top1 4/4, cosine>=0.9996). That "
    "spread is 48 layers of bf16 plus EP all-reduce/kernel differences, so "
    "5e-2 leaves ~1.9x headroom. The earlier 1e-2 was an estimate and is "
    "superseded. Graph correctness has its own tight gate: nano graph vs nano "
    "eager <= --graph-rel-l2-max (default 1e-5; observed 0). mean_abs (1.0) is "
    "scale-dependent and near-vacuous for logits in +-10..30; it is retained, "
    "unchanged, as a loose secondary gate. top1 (0.75, unchanged) is discrete "
    "and near-tie sensitive; top1_margin is reported so mismatches can be "
    "classified as bf16 artifacts vs bugs. top5_overlap is diagnostic."
)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(a, b, topk=5):
    """a: nano logits [V] f32; b: reference logits [V] f32."""
    import torch.nn.functional as F

    a = a.detach().float()
    b = b.detach().float()
    diff = (a - b).abs()
    # Least-squares gain: if nano's logits are a near-uniform rescale of the
    # reference's, rel_l2 overstates the discrepancy while argmax/top-k stay
    # intact.  Diagnostic only; it separates "systematic gain" from "noise".
    bb = float(b @ b)
    ls_scale = float((a @ b) / bb) if bb > 0 else 1.0
    rel_l2_scaled = float((a - ls_scale * b).norm() / b.norm().clamp_min(1e-12))
    top1 = float(a.argmax() == b.argmax())
    # A top1 mismatch whose reference margin (top1 - top2) is within bf16
    # error is a precision artifact, not a correctness bug.  Reported, never
    # gated: the 0.75 threshold stays as it was.
    top2 = b.topk(2).values
    top1_margin = float(top2[0] - top2[1])
    ka = set(a.topk(topk).indices.tolist())
    kb = set(b.topk(topk).indices.tolist())
    topk_overlap = len(ka & kb) / float(topk)
    return {
        "top1": top1,
        "top1_margin": top1_margin,
        "top5_overlap": topk_overlap,
        "mean_abs": diff.mean().item(),
        "max_abs": diff.max().item(),
        "cosine": F.cosine_similarity(a, b, dim=0).item(),
        "rel_l2": (diff.norm() / b.norm().clamp_min(1e-12)).item(),
        "ls_scale": ls_scale,
        "rel_l2_scaled": rel_l2_scaled,
        "nano_top1_token": int(a.argmax()),
        "ref_top1_token": int(b.argmax()),
    }


# ---------------------------------------------------------------------------
# Stage: nano
# ---------------------------------------------------------------------------

def run_nano(args):
    import torch
    from nanovllm.engine.llm_engine import LLMEngine
    from nanovllm.sampling_params import SamplingParams

    # The driver may have saved either a tensor or a bare list (older runs
    # did the latter); LLMEngine.add_request wants a list[int].
    prompt_ids = torch.load(os.path.join(args.out_dir, "prompt_ids.pt"),
                            map_location="cpu", weights_only=False)
    if torch.is_tensor(prompt_ids):
        prompt_ids = prompt_ids.tolist()
    else:
        prompt_ids = list(prompt_ids)
    mode = args.mode
    enforce_eager = (mode == "eager")

    engine = LLMEngine(
        args.model,
        tensor_parallel_size=args.tp,
        expert_parallel_size=args.ep,
        enforce_eager=enforce_eager,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_mem,
    )
    # Rank-0 peak after load + finalize + warmup/capture. Run with
    # --max-num-seqs 512 to include every graph tier when measuring the pool.
    print(f"[nano:{mode}] peak_allocated={torch.cuda.max_memory_allocated() / 1e9:.2f}GB "
          f"peak_reserved={torch.cuda.max_memory_reserved() / 1e9:.2f}GB")

    # Capture the logits the runner produces. Installed AFTER construction so
    # warmup/cudagraph capture are not recorded. `call()` resolves methods via
    # getattr(self, name), so an instance attribute shadows the class method on
    # the driver rank only; the worker ranks are unaffected.
    runner = engine.model_runner
    if mode == "graph":
        # Capture happens in ModelRunner.__init__; fail loudly here rather than
        # silently testing the eager path while claiming to test graphs.
        assert getattr(runner, "graphs", None), (
            "graph mode requested, but ModelRunner captured no CUDA graphs")
    graph_max_bs = int(getattr(runner, "graph_max_bs", 0))

    original_run_model = runner.run_model
    steps = []

    def wrapped_run_model(input_ids, positions, is_prefill):
        # Mirror ModelRunner.run_model's branch so the report can prove the
        # graph path was actually exercised, not merely captured.  Prefill is
        # eager by design; only decode can replay.
        uses_graph = (not is_prefill and not runner.enforce_eager
                      and input_ids.size(0) <= graph_max_bs)
        logits = original_run_model(input_ids, positions, is_prefill)
        steps.append({
            "input_ids": input_ids.detach().cpu().clone(),
            "positions": positions.detach().cpu().clone(),
            "is_prefill": bool(is_prefill),
            "used_graph": bool(uses_graph),
            # [num_returned_positions, vocab] -- 1 row for both a single-sequence
            # prefill and a decode step
            "logits": logits.detach().float().cpu().clone(),
        })
        return logits

    runner.run_model = wrapped_run_model

    sp = SamplingParams(temperature=1e-6, max_tokens=args.max_tokens,
                        ignore_eos=True)
    engine.add_request(prompt_ids, sp)
    assert not engine.failed, f"request rejected: {engine.failed}"

    n_steps = 0
    while not engine.is_finished():
        engine.step()
        n_steps += 1
        # prefill + one step per completion token; the +4 is headroom, not a
        # target -- a runaway loop here means the scheduler is not advancing
        if n_steps > args.max_tokens + 4:
            raise RuntimeError(f"generation did not finish after {n_steps} steps")

    assert steps, "no logits captured"

    n_replay = sum(1 for s in steps if s["used_graph"])
    if mode == "graph":
        assert n_replay > 0, (
            "graph mode requested, but every decode step took the eager "
            "branch; the gate would not be exercising replay")
    else:
        assert n_replay == 0, (
            "eager mode recorded a step on the graph replay branch")

    payload = {
        "mode": mode,
        "enforce_eager": enforce_eager,
        "tp": args.tp,
        "ep": args.ep,
        "model": args.model,
        "graph_max_bs": graph_max_bs,
        "n_graph_replay": n_replay,
        "steps": steps,
    }
    out = os.path.join(args.out_dir, f"nano_{mode}.pt")
    torch.save(payload, out)
    print(f"[nano:{mode}] captured {len(steps)} step(s), "
          f"{n_replay} graph replay(s) -> {out}")


# ---------------------------------------------------------------------------
# Stage: transformers reference
# ---------------------------------------------------------------------------

def _load_reference_model(args):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM

    print(f"[ref] transformers {transformers.__version__}, device_map={args.ref_device}")
    kwargs = dict(torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    if args.ref_device == "cpu":
        kwargs["device_map"] = "cpu"
    else:
        try:
            import accelerate  # noqa: F401  (device_map="auto" needs it)
        except ImportError:
            print("[ref] accelerate missing -- falling back to CPU")
            kwargs["device_map"] = "cpu"
        else:
            kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    model.eval()
    return model


def run_ref(args):
    import torch

    nano = torch.load(os.path.join(args.out_dir, f"nano_{args.mode}.pt"),
                      map_location="cpu", weights_only=False)
    steps = nano["steps"]

    # The whole gate assumes one sequence and one logits row per step (prefill
    # returns only the last position; each decode step produces one position).
    assert steps[0]["is_prefill"], "first captured step is not a prefill"
    assert all(s["logits"].shape[0] == 1 for s in steps), (
        "expected exactly one logits row per step, got "
        f"{[int(s['logits'].shape[0]) for s in steps]}")

    # Reconstruct the exact token sequence the model actually saw. For a single
    # sequence: prefill emits the whole prompt, each decode step emits one token.
    full_ids = torch.cat([s["input_ids"].reshape(-1) for s in steps], dim=0)
    sel_positions = []
    for s in steps:
        pos = s["positions"].reshape(-1)
        # prefill logits correspond to the last position of the chunk
        # (see ParallelLMHead.forward); decode logits to its single position
        sel_positions.append(int(pos[-1] if s["is_prefill"] else pos[0]))
    assert max(sel_positions) < full_ids.numel(), (
        f"position {max(sel_positions)} outside the {full_ids.numel()}-token sequence")

    model = _load_reference_model(args)
    with torch.inference_mode():
        out = model(input_ids=full_ids.unsqueeze(0).to(model.device),
                    use_cache=False)
        logits = out.logits[0]
        device = logits.device
        rows = logits[torch.tensor(sel_positions, device=device)].float().cpu()

    payload = {
        "mode": args.mode,
        "full_ids": full_ids,
        "sel_positions": sel_positions,
        "logits": rows,
    }
    path = os.path.join(args.out_dir, f"ref_{args.mode}.pt")
    torch.save(payload, path)
    print(f"[ref:{args.mode}] {len(sel_positions)} position(s), "
          f"seq_len={full_ids.numel()} -> {path}")


# ---------------------------------------------------------------------------
# Stage: compare
# ---------------------------------------------------------------------------

def run_compare(args):
    import torch

    nano = torch.load(os.path.join(args.out_dir, f"nano_{args.mode}.pt"),
                      map_location="cpu", weights_only=False)
    ref = torch.load(os.path.join(args.out_dir, f"ref_{args.mode}.pt"),
                     map_location="cpu", weights_only=False)
    steps = nano["steps"]
    ref_logits = ref["logits"]
    assert len(steps) == ref_logits.shape[0], (
        f"{len(steps)} nano steps vs {ref_logits.shape[0]} reference rows")

    rows = []
    for i, (step, r) in enumerate(zip(steps, ref_logits)):
        m = compute_metrics(step["logits"][0], r)
        m["step"] = i
        m["kind"] = "prefill" if step["is_prefill"] else "decode"
        m["tokens_in_step"] = int(step["input_ids"].numel())
        rows.append(m)

    n = len(rows)
    agg = {
        "top1": sum(r["top1"] for r in rows) / n,
        "top5_overlap": sum(r["top5_overlap"] for r in rows) / n,
        "mean_abs": sum(r["mean_abs"] for r in rows) / n,
        "max_abs": max(r["max_abs"] for r in rows),
        "cosine": sum(r["cosine"] for r in rows) / n,
        "rel_l2": sum(r["rel_l2"] for r in rows) / n,
        # Worst step: a single stale replay (e.g. one graph still reading the
        # capture-time input address) must fail the gate, and averaging over
        # max_tokens + 1 steps would dilute it.
        "rel_l2_max": max(r["rel_l2"] for r in rows),
        "min_top1_margin": min(r["top1_margin"] for r in rows),
        # Diagnostics: a near-constant gain means rel_l2 overstates the error
        # while argmax/top-k are unaffected.  Not gated.
        "ls_scale_mean": sum(r["ls_scale"] for r in rows) / n,
        "ls_scale_min": min(r["ls_scale"] for r in rows),
        "ls_scale_max": max(r["ls_scale"] for r in rows),
        "rel_l2_scaled_max": max(r["rel_l2_scaled"] for r in rows),
    }
    # rel_l2 is added; top1/cosine/mean_abs thresholds are unchanged.  See
    # CRITERION_NOTES at the top of this file for why 1e-2 and why worst-step.
    passed = (agg["top1"] >= args.top1_min
              and agg["cosine"] >= args.cos_min
              and agg["mean_abs"] <= args.mean_abs_max
              and agg["rel_l2_max"] <= args.rel_l2_max)

    mode = nano["mode"]
    print(f"\n===== MoE TP/EP logits gate: mode={mode} "
          f"(tp={nano['tp']}, ep={nano['ep']}, enforce_eager={nano['enforce_eager']}) =====")
    print(f"{'step':>4} {'kind':>8} {'tokens':>6} {'top1':>6} {'top5':>6} "
          f"{'mean|d|':>9} {'max|d|':>9} {'cosine':>9} {'relL2':>9}")
    for r in rows:
        print(f"{r['step']:>4} {r['kind']:>8} {r['tokens_in_step']:>6} "
              f"{r['top1']:>6.2f} {r['top5_overlap']:>6.2f} {r['mean_abs']:>9.4f} "
              f"{r['max_abs']:>9.4f} {r['cosine']:>9.6f} {r['rel_l2']:>9.6f}")
    print(f"{'ALL':>4} {'':>8} {'':>6} {agg['top1']:>6.2f} {agg['top5_overlap']:>6.2f} "
          f"{agg['mean_abs']:>9.4f} {agg['max_abs']:>9.4f} {agg['cosine']:>9.6f} "
          f"{agg['rel_l2']:>9.6f}")
    print(f"least-squares nano/ref gain: mean={agg['ls_scale_mean']:.4f} "
          f"[{agg['ls_scale_min']:.4f}, {agg['ls_scale_max']:.4f}]; "
          f"relL2 after removing it: max={agg['rel_l2_scaled_max']:.6f}")
    for r in rows:
        if r["top1"] == 0:
            # Diagnostic heuristic only, NOT a pass criterion: margins below
            # bf16 logit error (~1e-2) make an argmax flip expected.
            verdict = ("near-tie, likely bf16 artifact"
                       if r["top1_margin"] < 0.05
                       else "margin is wide -- investigate")
            print(f"note: step {r['step']} top1 mismatch, ref top1-top2 "
                  f"margin={r['top1_margin']:.4f} ({verdict})")
    print(f"PASS thresholds: top1>={args.top1_min}, cosine>={args.cos_min}, "
          f"mean|d|<={args.mean_abs_max}, relL2(worst step)<={args.rel_l2_max}")
    print(f"RESULT: {'PASS' if passed else 'FAIL'}")

    report = {
        "mode": mode,
        "tp": nano["tp"],
        "ep": nano["ep"],
        "enforce_eager": nano["enforce_eager"],
        "model": nano["model"],
        "graph_max_bs": nano.get("graph_max_bs", 0),
        "n_graph_replay": nano.get("n_graph_replay", 0),
        "steps": rows,
        "aggregate": agg,
        "thresholds": {
            "top1_min": args.top1_min,
            "cos_min": args.cos_min,
            "mean_abs_max": args.mean_abs_max,
            "rel_l2_max": args.rel_l2_max,
        },
        "criterion_notes": CRITERION_NOTES,
        "passed": passed,
    }
    path = os.path.join(args.out_dir, f"report_{mode}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"report -> {path}")
    return passed


def compare_nano_modes(args, modes):
    """Direct nano-graph vs nano-eager gate on identical inputs.

    This is the real CUDA-graph correctness check: same weights, same inputs,
    same kernels; only capture/replay differs.  A replay that reads
    capture-time addresses (or a MoE buffer that was not refreshed) fails here
    immediately, whereas the transformers comparison is dominated by bf16
    kernel/reference spread (~2e-2 on Qwen3-30B-A3B).

    Returns True when the comparison is not applicable (only one mode run).
    """
    import torch

    if not ("eager" in modes and "graph" in modes):
        print("[driver] eager-vs-graph comparison skipped (needs both modes)")
        return True

    def load(mode):
        return torch.load(os.path.join(args.out_dir, f"nano_{mode}.pt"),
                          map_location="cpu", weights_only=False)

    eager, graph = load("eager"), load("graph")
    assert len(eager["steps"]) == len(graph["steps"]), (
        f"eager captured {len(eager['steps'])} step(s), graph "
        f"{len(graph['steps'])}")

    rows = []
    for i, (se, sg) in enumerate(zip(eager["steps"], graph["steps"])):
        assert se["is_prefill"] == sg["is_prefill"], (
            f"step {i}: prefill/decode mismatch")
        assert torch.equal(se["input_ids"], sg["input_ids"]), (
            f"step {i}: eager and graph saw different input_ids")
        assert torch.equal(se["positions"], sg["positions"]), (
            f"step {i}: eager and graph saw different positions")
        m = compute_metrics(sg["logits"][0], se["logits"][0])
        m["step"] = i
        m["kind"] = "prefill" if se["is_prefill"] else "decode"
        m["used_graph"] = bool(sg.get("used_graph", False))
        rows.append(m)

    agg = {
        "rel_l2_max": max(r["rel_l2"] for r in rows),
        "max_abs": max(r["max_abs"] for r in rows),
        "top1": sum(r["top1"] for r in rows) / len(rows),
    }
    # 1e-5 is a tiny allowance for nondeterministic accumulation (e.g.
    # index_add_); the observed eager/graph difference is exactly 0.
    passed = agg["rel_l2_max"] <= args.graph_rel_l2_max

    print("\n===== nano eager vs nano graph (identical inputs) =====")
    for r in rows:
        print(f"{r['step']:>4} {r['kind']:>8} used_graph={int(r['used_graph'])} "
              f"relL2={r['rel_l2']:.3e} max|d|={r['max_abs']:.3e} "
              f"top1={r['top1']:.0f}")
    print(f"PASS threshold: relL2<={args.graph_rel_l2_max:g} "
          f"(same kernels; a larger difference means stale replay data)")
    print(f"RESULT: {'PASS' if passed else 'FAIL'}")

    report = {
        "steps": rows,
        "aggregate": agg,
        "thresholds": {"graph_rel_l2_max": args.graph_rel_l2_max},
        "passed": passed,
    }
    path = os.path.join(args.out_dir, "report_eager_vs_graph.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"report -> {path}")
    return passed


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _subprocess_args(args, stage, mode):
    cmd = [sys.executable, os.path.abspath(__file__), "--stage", stage,
           "--model", args.model, "--out-dir", args.out_dir,
           "--tp", str(args.tp), "--ep", str(args.ep),
           "--max-tokens", str(args.max_tokens),
           "--max-num-seqs", str(args.max_num_seqs),
           "--gpu-mem", str(args.gpu_mem),
           "--prompt-len", str(args.prompt_len),
           "--seed", str(args.seed),
           "--ref-device", args.ref_device,
           "--top1-min", str(args.top1_min),
           "--cos-min", str(args.cos_min),
           "--mean-abs-max", str(args.mean_abs_max),
           "--rel-l2-max", str(args.rel_l2_max)]
    if stage != "driver":
        cmd += ["--mode", mode]
    return cmd


def run_driver(args):
    import torch
    from transformers import AutoTokenizer

    os.makedirs(args.out_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    ids = tokenizer.encode(args.prompt, add_special_tokens=False)
    if not ids:
        raise RuntimeError("prompt tokenized to zero tokens")
    ids = ids[:args.prompt_len]
    # Save a tensor (canonical for torch.save); run_nano still accepts the bare
    # list written by older versions of this script.
    torch.save(torch.tensor(ids, dtype=torch.long),
               os.path.join(args.out_dir, "prompt_ids.pt"))
    print(f"[driver] prompt: {len(ids)} tokens -> "
          f"{os.path.join(args.out_dir, 'prompt_ids.pt')}")

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    results = {}
    for mode in modes:
        assert mode in ("eager", "graph"), f"unknown mode {mode!r}"
        print(f"\n[driver] ===== mode {mode} =====")
        subprocess.run(_subprocess_args(args, "nano", mode), check=True)
        subprocess.run(_subprocess_args(args, "ref", mode), check=True)
        args.mode = mode
        results[mode] = run_compare(args)

    # The tightest graph gate: nano replay vs nano eager on the same inputs.
    results["eager_vs_graph"] = compare_nano_modes(args, modes)

    print("\n[driver] summary:", {m: ("PASS" if p else "FAIL")
                                 for m, p in results.items()})
    return all(results.values())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", default="driver",
                   choices=["driver", "nano", "ref", "compare"])
    p.add_argument("--model", required=True)
    p.add_argument("--out-dir", default="/tmp/moe_tp_gate")
    p.add_argument("--modes", default="eager,graph",
                   help="comma-separated: eager,graph")
    p.add_argument("--mode", default=None, help="internal: single mode for sub-stages")
    p.add_argument("--tp", type=int, default=4)
    p.add_argument("--ep", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=4,
                   help="completion tokens (prefill + this many decode positions)")
    p.add_argument("--prompt-len", type=int, default=64,
                   help="truncate the fixed prompt to this many tokens")
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--max-num-seqs", type=int, default=1,
                   help="keep 1 so only small graph tiers are captured")
    p.add_argument("--gpu-mem", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=1234,
                   help="reserved; generation is effectively deterministic at "
                        "temperature=1e-6, so this seed is not consumed")
    p.add_argument("--ref-device", default="auto", choices=["auto", "cpu"],
                   help="transformers placement; auto needs accelerate")
    p.add_argument("--top1-min", type=float, default=0.75)
    p.add_argument("--cos-min", type=float, default=0.99)
    p.add_argument("--mean-abs-max", type=float, default=1.0)
    p.add_argument("--rel-l2-max", type=float, default=5e-2,
                   help="worst-step relative L2 vs the transformers reference. "
                        "Calibrated on Qwen3-30B-A3B TP=4/EP=4: observed "
                        "1.1e-2..2.6e-2, so 5e-2 leaves ~1.9x headroom")
    p.add_argument("--graph-rel-l2-max", type=float, default=1e-5,
                   help="nano graph vs nano eager on identical inputs; same "
                        "kernels, so the observed difference is 0 and any "
                        "larger value indicates stale replay data")
    return p


def main():
    args = build_parser().parse_args()
    if args.stage == "nano":
        assert args.mode is not None
        run_nano(args)
    elif args.stage == "ref":
        assert args.mode is not None
        run_ref(args)
    elif args.stage == "compare":
        assert args.mode is not None
        sys.exit(0 if run_compare(args) else 1)
    else:
        sys.exit(0 if run_driver(args) else 1)


if __name__ == "__main__":
    main()
