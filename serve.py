"""Plays one episode over stdin/stdout as JSON lines, so any agent harness
can drive the environment as a subprocess.

    python serve.py --seed 1

Output, one JSON object per line:
    {"type": "rules", "text": "..."}                 once, at the start
    {"type": "observation", "observation": {...}}    before every decision
    {"type": "result", "score": ..., "total_cost": ...}   at the end

Input, one JSON object per line:
    {"orders": [q0, q1, q2, q3]}     place this week's orders
    {"stop": true}                   give up; remaining weeks get zero orders

Closing stdin counts as stopping. An unreadable line counts as a malformed
order (all zeros) for that week.
"""

from __future__ import annotations

import argparse
import json
import sys

from supplychain import SupplyChainEnv, rules_text
from supplychain.scoring import anchors, score


def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    env = SupplyChainEnv(args.seed)
    emit({"type": "rules", "text": rules_text(env.cfg)})
    emit({"type": "observation", "observation": env.observe()})
    while not env.done:
        line = sys.stdin.readline()
        if not line:
            break
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            msg = {}
        if isinstance(msg, dict) and msg.get("stop"):
            break
        orders = msg.get("orders") if isinstance(msg, dict) else None
        res = env.step(orders)
        if not res.done:
            emit({"type": "observation", "observation": res.observation})
    env.finish()
    naive, floor = anchors(args.seed)
    emit({"type": "result", "seed": args.seed,
          "score": round(score(env.total_cost, naive, floor), 1),
          "total_cost": round(env.total_cost, 1),
          "handover_penalty": round(env.handover_penalty, 1),
          "invalid_actions": env.invalid_actions})


if __name__ == "__main__":
    main()
