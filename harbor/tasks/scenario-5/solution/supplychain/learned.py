"""A small trainable policy, used to show that the reward can be learned from.

The policy has nine numbers. It smooths the demand history at two speeds,
turns the two averages into a demand estimate, and has each stage order up to
a multiple of that estimate. It knows nothing about the demand model in the
rules and does not use the reference's forecast; `train.py` finds its nine
numbers from episode cost alone.

    alpha_fast, alpha_slow   smoothing speeds of the two demand averages
    gain_up, gain_down       how far to follow the fast average when it is
                             above / below the slow one
    z                        extra stock per unit of recent forecast error
    c0..c3                   weeks of demand each stage keeps at or below it
"""

from __future__ import annotations

import json
import math
import pathlib

from .policies import _cumulative_leads, _echelon_positions

PARAM_NAMES = ("alpha_fast", "alpha_slow", "gain_up", "gain_down", "z",
               "c0", "c1", "c2", "c3")
LOW = (0.01, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
HIGH = (1.0, 1.0, 3.0, 3.0, 6.0, 12.0, 14.0, 16.0, 20.0)

# Before training: no view on demand changes, no safety stock, and each
# stage covers exactly its own delivery time.
UNTRAINED = (0.5, 0.5, 0.0, 0.0, 0.0, 2.0, 4.0, 6.0, 9.0)

WEIGHTS_FILE = pathlib.Path(__file__).with_name("learned_weights.json")


def clip(params) -> tuple:
    return tuple(min(h, max(l, p)) for p, l, h in zip(params, LOW, HIGH))


def make_policy(params):
    a_fast, a_slow, g_up, g_down, z, *cover = clip(params)

    def policy(obs: dict) -> list:
        hist = obs["demand_history"]
        fast = slow = float(hist[0])
        err = 0.0
        for d in hist[1:]:
            err += 0.2 * (abs(d - fast) - err)
            fast += a_fast * (d - fast)
            slow += a_slow * (d - slow)
        gain = g_up if fast > slow else g_down
        level = max(0.0, slow + gain * (fast - slow))
        orders = []
        for pos, lead, c in zip(_echelon_positions(obs),
                                _cumulative_leads(obs), cover):
            target = c * level + z * err * math.sqrt(lead)
            orders.append(max(0, int(round(target - pos))))
        if obs["week"] >= obs["horizon"]:
            gap = obs["handover_requirement"] - (obs["system_stock"] + orders[-1])
            orders[-1] += max(0, int(math.ceil(gap)))
        return orders
    return policy


def load_trained():
    """The policy saved by train.py, or None if it has not been trained."""
    if not WEIGHTS_FILE.exists():
        return None
    data = json.loads(WEIGHTS_FILE.read_text())
    return make_policy([data["params"][k] for k in PARAM_NAMES])
