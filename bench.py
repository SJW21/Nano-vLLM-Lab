import argparse
import atexit
import json
import os
import tempfile
import time
from random import randint, seed
import torch
from nanovllm import LLM, SamplingParams


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="~/huggingface/Qwen3-0.6B/")
    parser.add_argument("--seqs", type=int, default=256)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--dp", type=int, default=1, help="data parallel size")
    parser.add_argument("--dplb-method", default="ROUND_ROBIN",
                        help="DP load balancing: ROUND_ROBIN or LEAST_REQUESTS")
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--async-scheduling", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="run-ahead scheduling (default on; --no-async-scheduling for the sync baseline)")
    parser.add_argument("--async-event-loop", action="store_true",
                        help="DP threaded event loop; requires --no-async-scheduling "
                             "(the two are mutually exclusive)")
    parser.add_argument("--arrival", type=int, default=0, metavar="N",
                        help="server-style: admit one request every N steps instead of one "
                             "offline batch (0 = offline; async scheduling's revenue case)")
    parser.add_argument("--arrival-ms", dest="arrival_ms", type=int, default=0, metavar="N",
                        help="DP only: admit one request every N milliseconds (0 = offline)")
    parser.add_argument("--rounds", type=int, default=1,
                        help="repeat the generate (same engine): F1/F2 long-run and engine "
                             "reuse; every round also prints the scheduler invariants")
    parser.add_argument("--max-input-len", type=int, default=1024)
    parser.add_argument("--max-output-len", type=int, default=1024)
    args = parser.parse_args()
    # the step-driven arrival loop needs LLM.step()/is_finished(), which DPLLMEngine
    # does not have; its DP counterpart admits on the wall clock instead
    if args.arrival and args.dp > 1:
        parser.error("--arrival (step-driven) has no DP counterpart; use --arrival-ms")
    if args.arrival_ms and args.dp <= 1:
        parser.error("--arrival-ms requires --dp > 1")
    if args.async_event_loop and args.async_scheduling:
        parser.error("--async-event-loop requires --no-async-scheduling: "
                     "async scheduling is not wired for the threaded event loop")
    return args


def report_invariants(llm, round_index):
    """F1/F2 + D4/D6, read after a round: nothing staged, no slot leaked, pools stable."""
    engine = llm._engine
    scheduler = engine.scheduler
    staged = (getattr(engine, "_pending", None) is not None,
              getattr(engine, "_inflight", None) is not None)
    pinned = getattr(engine.model_runner, "pinned_sampled", None)
    pinned_kib = sum(t.numel() * t.element_size() for t in pinned) / 1024 if pinned else 0
    print(f"[round {round_index}] invariants: num_inflight={scheduler.num_inflight} "
          f"waiting={len(scheduler.waiting)} running={len(scheduler.running)} "
          f"slot_free={len(scheduler.slot_free)}/{scheduler.max_num_seqs} staged={staged} "
          f"mem={torch.cuda.memory_allocated() / 2 ** 20:.0f}MiB pinned={pinned_kib:.0f}KiB",
          flush=True)


def report_dp_invariants(diag_dir, round_index):
    """Per-client drain state, read from the diag files the clients write at exit."""
    names = [name for name in sorted(os.listdir(diag_dir))
             if name.startswith("client_") and name.endswith(".json")]
    if not names:
        print(f"[round {round_index}] no client diag file under {diag_dir}", flush=True)
        return
    for name in names:
        with open(os.path.join(diag_dir, name), encoding="utf-8") as f:
            diag = json.load(f)
        inv = diag["invariants"]
        print(f"[round {round_index}] {name[:-5]} invariants: num_inflight={inv['num_inflight']} "
              f"waiting={inv['waiting']} running={inv['running']} "
              f"slot_free={inv['slot_free']}/{inv['max_num_seqs']} "
              f"staged=({inv['staged_pending']}, {inv['staged_inflight']}) "
              f"mem={diag['peak_alloc_mib']:.0f}MiB", flush=True)


def run_offline(llm, prompt_token_ids, sampling_params):
    t = time.time()
    outputs = llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    return time.time() - t, len(outputs)


def run_arrival(llm, prompt_token_ids, sampling_params, interval):
    """Admit one request every `interval` steps, polling only is_finished() -- the loop
    a server would run (which counts in-flight batches, so nothing is left staged)."""
    added = steps = 0
    t = time.time()
    while added < len(prompt_token_ids) or not llm.is_finished():
        if added < len(prompt_token_ids) and steps % interval == 0:
            llm.add_request(prompt_token_ids[added], sampling_params[added])
            added += 1
        llm.step()
        steps += 1
    return time.time() - t, steps, added


def run_arrival_dp(llm, prompts, params, interval_ms):
    """DP arrival driver: admit one request every `interval_ms` of wall clock and poll
    the controller's result socket (DPLLMEngine has no step/is_finished API)."""
    engine = llm._engine
    results = {}
    engine.dp_controller.Res_Budget = results
    n = len(prompts)
    added = 0
    t0 = time.time()
    t_admit = None
    while added < n or len(results) < n - len(engine.failed):
        if added < n and (time.time() - t0) * 1000 >= added * interval_ms:
            llm.add_request(prompts[added], params[added])
            added += 1
            if added == n:
                t_admit = time.time() - t0
        engine.dp_controller.res_recv_from_client()
        time.sleep(0.0005)
    dt = time.time() - t0
    return dt, len(results), t_admit


def run_dp(args, llm, prompt_token_ids, sampling_params, total_tokens, diag_dir):
    engine = llm._engine
    mode = "arrival" if args.arrival_ms else "offline"
    for round_index in range(1, args.rounds + 1):
        t0 = time.time()
        if mode == "arrival":
            dt, n_out, t_admit = run_arrival_dp(llm, prompt_token_ids, sampling_params,
                                                args.arrival_ms)
        else:
            dt, n_out = run_offline(llm, prompt_token_ids, sampling_params)
            t_admit = None
        total_wall = time.time() - t0
        print(f"RESULT mode={mode} round={round_index} "
              f"throughput_tok_s={total_tokens / dt:.2f} elapsed_s={dt:.2f} "
              f"total_wall_s={total_wall:.2f} tokens={total_tokens} "
              f"outputs={n_out}/{args.seqs} rejected={len(engine.failed)} "
              f"admit_span_s={(t_admit if t_admit is not None else -1):.3f} "
              f"admit_ratio={(t_admit / dt if t_admit else -1):.3f}", flush=True)
    # the clients write their diag file in DPClient.exit(), which the controller joins
    # in engine.exit(): exit explicitly here, otherwise atexit runs after main() returns
    atexit.unregister(engine.exit)
    engine.exit()
    report_dp_invariants(diag_dir, args.rounds)


def main():
    args = parse_args()
    seed(0)
    num_seqs = args.seqs

    path = os.path.expanduser(args.path)
    diag_dir = tempfile.mkdtemp(prefix="nanovllm-dp-diag-") if args.dp > 1 else None
    if diag_dir:
        # spawn'ed clients inherit the env: this must be set before the engine is built.
        # invariants only -- a revenue run must not pay for the per-row trace
        os.environ["NANOVLLM_DP_DIAG"] = diag_dir
        os.environ["NANOVLLM_DP_DIAG_MODE"] = "invariants"
    llm = LLM(path, enforce_eager=args.eager, tensor_parallel_size=args.tp,
              data_parallel_size=args.dp, dplb_method=args.dplb_method,
              max_model_len=4096, async_scheduling=args.async_scheduling,
              async_event_loop=args.async_event_loop)

    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, args.max_input_len))]
                        for _ in range(num_seqs)]
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True,
                                      max_tokens=randint(100, args.max_output_len))
                       for _ in range(num_seqs)]

    llm.generate(["Benchmark: "], SamplingParams(), use_tqdm=False)
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    if args.dp > 1:
        run_dp(args, llm, prompt_token_ids, sampling_params, total_tokens, diag_dir)
    elif args.arrival:
        t, steps, added = run_arrival(llm, prompt_token_ids, sampling_params, args.arrival)
        print(f"Arrival: 1 req / {args.arrival} steps, {added} reqs, {steps} steps")
        print(f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {total_tokens / t:.2f}tok/s")
        report_invariants(llm, 1)
    else:
        for round_index in range(1, args.rounds + 1):
            t, _ = run_offline(llm, prompt_token_ids, sampling_params)
            label = "Total" if args.rounds == 1 else f"[round {round_index}/{args.rounds}] Total"
            print(f"{label}: {total_tokens}tok, Time: {t:.2f}s, "
                  f"Throughput: {total_tokens / t:.2f}tok/s")
            report_invariants(llm, round_index)


if __name__ == "__main__":
    main()
