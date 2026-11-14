"""Scripted policies: baselines, the reference run and the scoring anchors.

Every policy except `perfect_foresight` sees only the observation dict, i.e.
exactly what the agent sees.
"""

from __future__ import annotations

import math

from .env import N
from .forecast import demand_quantiles

# Service levels of the reference policy, one per stage (retailer first):
# the probability that stock at or below the stage covers demand over its
# cumulative lead time. Found by search on seeds 1000-1149 and checked on
# seeds 1150-1249; none of these are evaluation seeds.
REFERENCE_SERVICE = (0.98, 0.97, 0.97, 0.97)


def _echelon_positions(obs: dict) -> list:
    """Echelon inventory position of each stage: all stock at or below the
    stage, plus everything on its way to it, minus customer backlog."""
    pos, running = [], -obs["customer_backlog"]
    for s in obs["stages"]:
        running += s["on_hand"] + sum(s["arriving"])
        pos.append(running + s["awaiting_from_supplier"])
    return pos


def _cumulative_leads(obs: dict) -> list:
    out, total = [], 0
    for s in obs["stages"]:
        total += s["lead_time"]
        out.append(total)
    return out


def do_nothing(obs: dict) -> list:
    return [0] * N


def pass_through(obs: dict) -> list:
    """Naive baseline: every stage orders what customers just bought."""
    return [obs["demand_this_week"]] * N


def reference(obs: dict, service: tuple | None = None) -> list:
    """Echelon base-stock policy driven by a Bayesian demand forecast.

    The forecast (see forecast.py) infers from the demand history which of
    the patterns described in the rules is unfolding. Each stage then orders
    up to a high quantile of forecast demand over its cumulative lead time,
    counting everything already at or on its way to that stage and the
    stages below it.
    """
    service = service or REFERENCE_SERVICE
    orders = []
    for pos, lead, p in zip(_echelon_positions(obs), _cumulative_leads(obs),
                            service):
        target = demand_quantiles(obs["demand_history"], lead, [p])[0]
        orders.append(max(0, int(round(target - pos))))
    if obs["week"] >= obs["horizon"]:
        gap = obs["handover_requirement"] - (obs["system_stock"] + orders[-1])
        orders[-1] += max(0, int(math.ceil(gap)))
    return orders


def make_perfect_foresight(env):
    """Scoring anchor only: a planner that knows future demand exactly.
    It is not available to agents and is not the reference run."""
    def policy(obs: dict) -> list:
        t = obs["week"]                      # 1-based; env._demand[t] is next week
        orders = []
        for pos, lead in zip(_echelon_positions(obs), _cumulative_leads(obs)):
            target = sum(env._demand[t:t + lead])
            orders.append(max(0, target - pos))
        if t >= obs["horizon"]:
            gap = obs["handover_requirement"] - (obs["system_stock"] + orders[-1])
            orders[-1] += max(0, int(math.ceil(gap)))
        return orders
    return policy


def stop_early(after_weeks: int):
    """Plays the reference for a few weeks, then stops ordering."""
    def policy(obs: dict) -> list:
        return reference(obs) if obs["week"] <= after_weeks else [0] * N
    return policy
