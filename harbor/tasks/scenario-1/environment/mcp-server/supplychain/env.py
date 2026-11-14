"""Delayed supply chain environment (a centralised four-stage "beer game").

The agent runs a serial supply chain:

    factory (3) -> distributor (2) -> wholesaler (1) -> retailer (0) -> customers

Every week it decides how much each stage orders from the stage above it.
Goods take several weeks to arrive, customer demand is noisy and shifts over
time, and the agent pays for stock it holds and for customer orders it cannot
fill. Everything needed to act well is in the observation; nothing is hidden
except future demand.

Only the Python standard library is used.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

STAGE_NAMES = ("retailer", "wholesaler", "distributor", "factory")
N = len(STAGE_NAMES)


@dataclass(frozen=True)
class Config:
    horizon: int = 52
    # Weeks between a shipment leaving the upstream stage and arriving.
    # For the factory this is its production time.
    lead_times: tuple = (2, 2, 2, 3)
    # Cost per unit of stock on hand per week. Stock is cheaper upstream.
    holding_cost: tuple = (1.0, 0.75, 0.5, 0.25)
    # Cost per unit of unfilled customer demand per week.
    backlog_cost: float = 10.0
    max_order: int = 200
    factory_capacity: int = 80
    # Handover rule: the chain must be left with enough stock in the system
    # to cover `sum(lead_times)` weeks of recent demand.
    handover_window: int = 8


@dataclass
class StepResult:
    observation: dict
    cost: float          # cost incurred in the week just simulated
    done: bool
    info: dict = field(default_factory=dict)


def sample_level(rng: random.Random, length: int) -> tuple:
    """Draw one demand scenario: returns (base level, level per week, noise cv).

    This is the demand model described to the agent in the rules: a starting
    level, one lasting change (step up, step down or ramp) and, half the time,
    a seasonal wave or a short promotion on top.
    """
    base = rng.uniform(15, 25)
    level = [base] * length
    # Every scenario has one lasting change of level, and half of them also
    # have a temporary disturbance on top.
    patterns = [rng.choice(["step_up", "step_down", "ramp"])]
    if rng.random() < 0.5:
        patterns.append(rng.choice(["seasonal", "promo"]))
    for p in patterns:
        start = rng.randint(8, 34)
        if p == "step_up":
            m = rng.uniform(1.4, 1.9)
            for t in range(start, length):
                level[t] *= m
        elif p == "step_down":
            m = rng.uniform(0.45, 0.7)
            for t in range(start, length):
                level[t] *= m
        elif p == "ramp":
            m = rng.uniform(1.4, 1.8)
            dur = rng.randint(8, 16)
            for t in range(start, length):
                level[t] *= 1 + (m - 1) * min(1.0, (t - start) / dur)
        elif p == "seasonal":
            amp = rng.uniform(0.25, 0.4)
            period = rng.randint(13, 26)
            phase = rng.uniform(0, 2 * math.pi)
            for t in range(length):
                level[t] *= 1 + amp * math.sin(2 * math.pi * t / period + phase)
        elif p == "promo":
            m = rng.uniform(1.8, 2.4)
            dur = rng.randint(3, 4)
            for t in range(start, min(length, start + dur)):
                level[t] *= m
            for t in range(start + dur, min(length, start + dur + 2)):
                level[t] *= 0.6
    cv = rng.uniform(0.15, 0.3)
    return base, level, cv


def generate_demand(seed: int, length: int) -> list:
    """Weekly demand for a scenario: the level from `sample_level` plus
    normally distributed noise, rounded and floored at zero."""
    rng = random.Random(seed * 7919 + 13)
    _, level, cv = sample_level(rng, length)
    return [max(0, int(round(rng.gauss(mu, cv * mu)))) for mu in level]


def _clean_order(x, cap: int):
    """Return (order, was_valid). Anything malformed becomes an order of 0."""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return 0, False
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return 0, False
    if x < 0:
        return 0, False
    return min(int(x), cap), True


class SupplyChainEnv:
    def __init__(self, seed: int = 0, config: Config | None = None):
        self.cfg = config or Config()
        self.seed = seed
        tail = sum(self.cfg.lead_times) + 2
        self._demand = generate_demand(seed, self.cfg.horizon + tail)
        self.reset()

    # ------------------------------------------------------------------ API

    def reset(self) -> dict:
        c = self.cfg
        # Start from a calm, balanced chain sized for the opening demand level.
        mu0 = max(1, int(round(sum(self._demand[:4]) / 4)))
        self.on_hand = [mu0] + [mu0 // 2] * (N - 1)
        self.pipeline = [[mu0] * L for L in c.lead_times]
        self.owed = [0] * (N - 1)   # owed[i]: units stage i+1 still owes stage i
        self.backlog = 0            # unfilled customer demand
        self.t = 1
        self.total_cost = 0.0
        self.handover_penalty = 0.0
        self.done = False
        self.invalid_actions = 0
        self.demand_history = []
        self.order_history = []
        self._begin_week()
        return self.observe()

    def step(self, orders) -> StepResult:
        """Place this week's orders (one per stage, retailer first)."""
        if self.done:
            raise RuntimeError("episode is finished; call reset()")
        c = self.cfg
        if not isinstance(orders, (list, tuple)) or len(orders) != N:
            orders, valid = [0] * N, [False] * N
        else:
            pairs = [_clean_order(x, c.max_order) for x in orders]
            orders, valid = [p[0] for p in pairs], [p[1] for p in pairs]
        self.invalid_actions += valid.count(False)
        orders[N - 1] = min(orders[N - 1], c.factory_capacity)
        self.order_history.append(list(orders))

        # Stages 0..2 order from the stage above, which ships what it can now
        # and keeps the rest as a backorder that ships when stock arrives.
        for i in range(N - 1):
            self.owed[i] += orders[i]
            shipped = min(self.on_hand[i + 1], self.owed[i])
            self.on_hand[i + 1] -= shipped
            self.owed[i] -= shipped
            self.pipeline[i][-1] += shipped
        # The factory produces from unlimited raw material, up to capacity.
        self.pipeline[N - 1][-1] += orders[N - 1]

        if self.t >= c.horizon:
            self.done = True
            self.handover_penalty = self._handover_penalty()
            self.total_cost += self.handover_penalty
            return StepResult(self.observe(), self.handover_penalty, True,
                              {"handover_penalty": self.handover_penalty})
        self.t += 1
        week_cost = self._begin_week()
        return StepResult(self.observe(), week_cost, False)

    def finish(self) -> float:
        """Run out the remaining weeks with zero orders (used when an agent
        stops early or crashes). Returns the final total cost."""
        while not self.done:
            self.step([0] * N)
        return self.total_cost

    # ------------------------------------------------------------ internals

    def _begin_week(self) -> float:
        c = self.cfg
        for i in range(N):
            self.on_hand[i] += self.pipeline[i].pop(0)
            self.pipeline[i].append(0)
        d = self._demand[self.t - 1]
        self.demand_history.append(d)
        need = self.backlog + d
        served = min(self.on_hand[0], need)
        self.on_hand[0] -= served
        self.backlog = need - served
        cost = sum(h * s for h, s in zip(c.holding_cost, self.on_hand))
        cost += c.backlog_cost * self.backlog
        self.total_cost += cost
        return cost

    def system_stock(self) -> int:
        """All physical stock in the chain, minus what customers are owed."""
        return (sum(self.on_hand) + sum(sum(p) for p in self.pipeline)
                - self.backlog)

    def handover_requirement(self) -> float:
        w = self.demand_history[-self.cfg.handover_window:]
        return sum(self.cfg.lead_times) * (sum(w) / len(w))

    def _handover_penalty(self) -> float:
        short = max(0.0, self.handover_requirement() - self.system_stock())
        return self.cfg.backlog_cost * short

    def observe(self) -> dict:
        c = self.cfg
        stages = []
        for i in range(N):
            stages.append({
                "stage": i,
                "name": STAGE_NAMES[i],
                "on_hand": self.on_hand[i],
                # arriving[k] reaches this stage in k+1 weeks
                "arriving": list(self.pipeline[i]),
                "awaiting_from_supplier": self.owed[i] if i < N - 1 else 0,
                "owed_to_downstream": self.owed[i - 1] if i > 0 else self.backlog,
                "lead_time": c.lead_times[i],
                "holding_cost": c.holding_cost[i],
            })
        return {
            "week": self.t,
            "horizon": c.horizon,
            "done": self.done,
            "customer_backlog": self.backlog,
            "demand_this_week": self.demand_history[-1],
            "demand_history": list(self.demand_history),
            "order_history": [list(o) for o in self.order_history],
            "stages": stages,
            "total_cost_so_far": round(self.total_cost, 2),
            "backlog_cost": c.backlog_cost,
            "max_order": c.max_order,
            "factory_capacity": c.factory_capacity,
            "handover_requirement": round(self.handover_requirement(), 1),
            "system_stock": self.system_stock(),
            "invalid_actions": self.invalid_actions,
        }


RULES = """\
You run a four-stage supply chain for {horizon} weeks:
factory (stage 3) -> distributor (2) -> wholesaler (1) -> retailer (0) -> customers.

Each week, in this order:
1. Shipments due this week arrive at each stage.
2. Customer demand hits the retailer. It is filled from retailer stock;
   anything unfilled becomes customer backlog and is filled later.
3. You are charged: stock on hand at each stage times its holding cost
   ({holding} per unit per week for stages 0-3), plus {backlog} per unit of
   customer backlog. Stock in transit is free.
4. You choose four order quantities, one per stage. Stages 0-2 order from the
   stage above. That stage ships what it has immediately and keeps the rest
   as a backorder ("awaiting_from_supplier") which ships automatically when
   its own stock arrives. Orders cannot be cancelled. The factory's order is
   a production order, limited to {capacity} per week.
   Shipments take {leads} weeks to reach stages 0-3 respectively.

Demand model. Each week's demand is drawn around an underlying level, with
noise whose standard deviation is 15-30% of the level (fixed per episode).
- The level starts between 15 and 25.
- Once per episode, starting between weeks 9 and 35, it changes for good:
  a step up (x1.4-1.9), a step down (x0.45-0.7), or a ramp that climbs to
  x1.4-1.8 over 8-16 weeks. You are not told which or when.
- In half of all episodes there is also one temporary effect: either a
  seasonal wave present all episode (+/-25-40%, period 13-26 weeks), or a
  promotion starting between weeks 9 and 35 (x1.8-2.4 for 3-4 weeks, then
  x0.6 for two weeks).

Handover rule: after the last week, total stock in the system (on hand plus
in transit, minus customer backlog) must be at least "handover_requirement"
(= {total_lead} weeks of average demand over the last {window} weeks). Each
unit short costs {backlog}.

Orders must be whole numbers from 0 to {max_order}. A malformed order counts
as 0. Your goal is the lowest total cost.
"""


def rules_text(cfg: Config | None = None) -> str:
    c = cfg or Config()
    return RULES.format(
        horizon=c.horizon, holding=", ".join(str(h) for h in c.holding_cost),
        backlog=c.backlog_cost, capacity=c.factory_capacity,
        leads=", ".join(str(x) for x in c.lead_times),
        total_lead=sum(c.lead_times), window=c.handover_window,
        max_order=c.max_order)
