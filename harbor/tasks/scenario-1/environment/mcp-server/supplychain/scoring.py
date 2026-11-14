"""Turning total cost into a 0-100 score.

For each scenario (seed) two anchors are computed on the same demand:

    naive  = cost of the pass-through baseline (order what was just sold)
    floor  = cost of a planner with perfect knowledge of future demand

    score = 100 * (naive - agent) / (naive - floor), clipped to [0, 100]

So doing no better than the naive rule scores 0, and 100 would require
knowing the future. A suite score pools the costs over all its scenarios
before applying the same formula, so one easy scenario cannot dominate.
"""

from __future__ import annotations

from .env import Config, SupplyChainEnv
from . import policies


def run_policy(policy, seed: int, config: Config | None = None) -> SupplyChainEnv:
    env = SupplyChainEnv(seed, config)
    obs = env.observe()
    while not env.done:
        obs = env.step(policy(obs)).observation
    return env


def anchors(seed: int, config: Config | None = None) -> tuple:
    naive = run_policy(policies.pass_through, seed, config).total_cost
    env = SupplyChainEnv(seed, config)
    pf = policies.make_perfect_foresight(env)
    obs = env.observe()
    while not env.done:
        obs = env.step(pf(obs)).observation
    return naive, env.total_cost


def score(agent_cost: float, naive: float, floor: float) -> float:
    span = naive - floor
    if span <= 0:
        return 0.0
    return max(0.0, min(100.0, 100.0 * (naive - agent_cost) / span))


def suite_score(costs: dict, config: Config | None = None) -> dict:
    """costs: {seed: total_cost}. Returns per-seed and pooled scores."""
    per_seed, tot_a, tot_n, tot_f = {}, 0.0, 0.0, 0.0
    for seed, cost in costs.items():
        naive, floor = anchors(seed, config)
        per_seed[seed] = {"cost": round(cost, 1), "naive": round(naive, 1),
                          "floor": round(floor, 1),
                          "score": round(score(cost, naive, floor), 1)}
        tot_a, tot_n, tot_f = tot_a + cost, tot_n + naive, tot_f + floor
    return {"score": round(score(tot_a, tot_n, tot_f), 1),
            "total_cost": round(tot_a, 1), "per_seed": per_seed}
