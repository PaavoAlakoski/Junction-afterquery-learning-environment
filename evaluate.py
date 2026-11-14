"""One-command evaluation.

    python evaluate.py                 reference run + all baselines
    python evaluate.py --agent claude  also run Claude three times (needs ANTHROPIC_API_KEY)
    python evaluate.py --agent llm     same, for any OpenAI-compatible endpoint

Baseline results are written to results/baselines.json. Each agent evaluation
is written to its own timestamped file, results/agent-<model>-<time>.json, so
paid runs and their transcripts are never overwritten.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import time
from concurrent.futures import ThreadPoolExecutor

from supplychain import SupplyChainEnv, learned, policies
from supplychain.scoring import suite_score

DEFAULT_SEEDS = [1, 2, 3, 4, 5]
STOP_EARLY_AFTER = 10


def run_episode(make_policy, seed) -> tuple:
    """Returns (total_cost, extras). If the policy raises, the rest of the
    episode is played with zero orders."""
    env = SupplyChainEnv(seed)
    policy = make_policy(seed)
    obs = env.observe()
    extras = {}
    try:
        while not env.done:
            obs = env.step(policy(obs)).observation
    except Exception as e:  # agent crashed: count it as stopping
        extras["errors"] = repr(e)
        env.finish()
    if env.invalid_actions:
        extras["invalid_actions"] = env.invalid_actions
    if hasattr(policy, "transcript"):
        extras["transcripts"] = policy.transcript
    return env.total_cost, extras


def run_suite(make_policy, seeds, workers: int = 1) -> dict:
    """make_policy(seed) -> callable(obs) -> orders. Scenarios are
    independent, so `workers` > 1 plays them in parallel."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda s: run_episode(make_policy, s), seeds))
    out = suite_score({s: r[0] for s, r in zip(seeds, results)})
    for seed, (_, extras) in zip(seeds, results):
        for key, value in extras.items():
            out.setdefault(key, {})[seed] = value
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    ap.add_argument("--agent", choices=["claude", "llm"], help="also run an agent")
    ap.add_argument("--workers", type=int, default=5,
                    help="scenarios played in parallel during agent runs")
    ap.add_argument("--runs", type=int, default=3, help="agent runs")
    ap.add_argument("--out", help="output file (default: see above)")
    args = ap.parse_args()

    rows = {
        "reference": run_suite(lambda s: policies.reference, args.seeds),
        "naive (pass-through)": run_suite(lambda s: policies.pass_through, args.seeds),
        "do nothing": run_suite(lambda s: policies.do_nothing, args.seeds),
        f"stop after week {STOP_EARLY_AFTER}": run_suite(
            lambda s: policies.stop_early(STOP_EARLY_AFTER), args.seeds),
    }
    # The small policy from train.py, before and after training (stretch goal).
    if learned.load_trained():
        rows["small policy, untrained"] = run_suite(
            lambda s: learned.make_policy(learned.UNTRAINED), args.seeds)
        rows["small policy, trained"] = run_suite(
            lambda s: learned.load_trained(), args.seeds)
    if args.agent:
        if args.agent == "claude":
            from supplychain.claude_agent import ClaudeAgent as Agent
        else:
            from supplychain.llm_agent import LLMAgent as Agent
        for i in range(1, args.runs + 1):
            print(f"running agent, run {i}/{args.runs} ...", flush=True)
            rows[f"agent run {i}"] = run_suite(lambda s: Agent(), args.seeds,
                                               args.workers)
        if args.agent == "claude":
            from supplychain.claude_agent import usage_totals as u
            print(f"Claude usage: {u['calls']} calls, {u['input_tokens']:,} "
                  f"input tokens, {u['output_tokens']:,} output tokens, "
                  f"about ${u['usd']:.2f}")

    print(f"\nSeeds: {args.seeds}\n")
    print(f"{'policy':<24}{'score':>8}{'total cost':>14}   per-seed scores")
    for name, r in rows.items():
        per = "  ".join(f"{v['score']:5.1f}" for v in r["per_seed"].values())
        print(f"{name:<24}{r['score']:>8.1f}{r['total_cost']:>14,.0f}   {per}")
        if "errors" in r:
            print(f"    errors: {r['errors']}")

    if args.out:
        out = pathlib.Path(args.out)
    elif args.agent:
        model = (os.environ.get("CLAUDE_MODEL", "claude-haiku-5-5")
                 if args.agent == "claude" else os.environ.get("LLM_MODEL", "llm"))
        stamp = time.strftime("%Y%m%d-%H%M%S")
        out = pathlib.Path(f"results/agent-{model.replace('/', '_')}-{stamp}.json")
    else:
        out = pathlib.Path("results/baselines.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"seeds": args.seeds, "results": rows}, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
