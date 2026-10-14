import argparse
import os
import time
from random import randint, seed
import torch
from nanovllm import LLM, SamplingParams
# from vllm import LLM, SamplingParams


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="~/huggingface/Qwen3-0.6B/")
    parser.add_argument("--seqs", type=int, default=256)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--async-scheduling", dest="async_scheduling", action="store_true",
                        help="run-ahead scheduling (see Devlog/async_scheduling_plan.md)")
    parser.add_argument("--arrival", type=int, default=0, metavar="N",
                        help="server-style: admit one request every N steps instead of one "
                             "offline batch (0 = offline; async scheduling's revenue case)")
    parser.add_argument("--rounds", type=int, default=1,
                        help="repeat the generate (same engine): F1/F2 long-run and engine "
                             "reuse; every round also prints the scheduler invariants")
    parser.add_argument("--max-input-len", type=int, default=1024)
    parser.add_argument("--max-output-len", type=int, default=1024)
    return parser.parse_args()


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


def run_offline(llm, prompt_token_ids, sampling_params):
    t = time.time()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    return time.time() - t


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


def main():
    args = parse_args()
    seed(0)
    num_seqs = args.seqs

    path = os.path.expanduser(args.path)
    llm = LLM(path, enforce_eager=args.eager, tensor_parallel_size=args.tp,
              max_model_len=4096, async_scheduling=args.async_scheduling)

    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, args.max_input_len))]
                        for _ in range(num_seqs)]
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True,
                                      max_tokens=randint(100, args.max_output_len))
                       for _ in range(num_seqs)]
    # uncomment the following line for vllm
    # prompt_token_ids = [dict(prompt_token_ids=p) for p in prompt_token_ids]

    llm.generate(["Benchmark: "], SamplingParams())
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    if args.arrival:
        t, steps, added = run_arrival(llm, prompt_token_ids, sampling_params, args.arrival)
        print(f"Arrival: 1 req / {args.arrival} steps, {added} reqs, {steps} steps")
        print(f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {total_tokens / t:.2f}tok/s")
        report_invariants(llm, 1)
    else:
        for round_index in range(1, args.rounds + 1):
            t = run_offline(llm, prompt_token_ids, sampling_params)
            label = "Total" if args.rounds == 1 else f"[round {round_index}/{args.rounds}] Total"
            print(f"{label}: {total_tokens}tok, Time: {t:.2f}s, "
                  f"Throughput: {total_tokens / t:.2f}tok/s")
            report_invariants(llm, round_index)


if __name__ == "__main__":
    main()
