"""Train the small policy in supplychain/learned.py from episode cost alone.

    python train.py

Method: cross-entropy search. Each generation samples candidate settings of
the policy's nine numbers, plays every candidate on the training scenarios,
keeps the cheapest quarter and samples the next generation around them.

Scenarios used (none are evaluation seeds, and none were used to tune the
reference):
    training    seeds 2000-2059
    held out    seeds 3000-3039   scored after every generation, never trained on

Writes the trained numbers to supplychain/learned_weights.json and the
learning curve to results/training.json. Deterministic: same output every run.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import time
from concurrent.futures import ProcessPoolExecutor

from supplychain import learned
from supplychain.scoring import anchors, run_policy, score

TRAIN_SEEDS = list(range(2000, 2060))
HELD_OUT_SEEDS = list(range(3000, 3040))
EVAL_SEEDS = [1, 2, 3, 4, 5]


def total_cost(args) -> float:
    params, seeds = args
    policy = learned.make_policy(params)
    return sum(run_policy(policy, s).total_cost for s in seeds)


def anchor_totals(seeds) -> tuple:
    pairs = [anchors(s) for s in seeds]
    return sum(p[0] for p in pairs), sum(p[1] for p in pairs)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--generations", type=int, default=30)
    ap.add_argument("--population", type=int, default=24)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    rng = random.Random(7)
    start = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        sets = {"train": TRAIN_SEEDS, "held_out": HELD_OUT_SEEDS,
                "evaluation": EVAL_SEEDS}
        anch = dict(zip(sets, pool.map(anchor_totals, sets.values())))

        def scores(params) -> dict:
            costs = pool.map(total_cost, [(params, s) for s in sets.values()])
            return {name: round(score(c, *anch[name]), 1)
                    for name, c in zip(sets, costs)}

        mean = list(learned.UNTRAINED)
        spread = [(h - l) / 6 for l, h in zip(learned.LOW, learned.HIGH)]
        curve = [{"generation": 0, "episodes": 0, **scores(mean)}]
        print(f"gen  0  {curve[0]}", flush=True)
        n_elite = max(2, args.population // 4)
        episodes = 0
        for gen in range(1, args.generations + 1):
            pop = [learned.clip([rng.gauss(m, s) for m, s in zip(mean, spread)])
                   for _ in range(args.population)]
            pop.append(tuple(mean))
            costs = list(pool.map(total_cost, [(p, TRAIN_SEEDS) for p in pop]))
            episodes += len(pop) * len(TRAIN_SEEDS)
            elite = [p for _, p in sorted(zip(costs, pop))[:n_elite]]
            for i in range(len(mean)):
                vals = [p[i] for p in elite]
                mean[i] = sum(vals) / len(vals)
                sd = (sum((v - mean[i]) ** 2 for v in vals) / len(vals)) ** 0.5
                floor = (learned.HIGH[i] - learned.LOW[i]) * 0.01
                spread[i] = max(floor, sd)
            curve.append({"generation": gen, "episodes": episodes, **scores(mean)})
            print(f"gen {gen:2d}  {curve[-1]}", flush=True)

    params = dict(zip(learned.PARAM_NAMES, (round(m, 4) for m in learned.clip(mean))))
    learned.WEIGHTS_FILE.write_text(json.dumps({
        "params": params, "trained_on": "seeds 2000-2059",
        "episodes": episodes, "final": curve[-1]}, indent=2))
    out = pathlib.Path("results/training.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "untrained": dict(zip(learned.PARAM_NAMES, learned.UNTRAINED)),
        "trained": params, "curve": curve}, indent=2))
    print(f"\nTrained in {time.time() - start:.0f}s over {episodes:,} episodes.")
    print(f"Wrote {learned.WEIGHTS_FILE} and {out}")


if __name__ == "__main__":
    main()
