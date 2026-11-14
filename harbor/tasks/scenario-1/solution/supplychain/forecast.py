"""Bayesian demand forecast used by the reference policy.

The rules tell the agent how demand is generated. This module turns that
description into a forecast: it draws a few thousand candidate scenarios from
the stated demand model, weighs each by how well it explains the demand seen
so far, and reads off the distribution of demand over the coming weeks.

It uses only the demand history, which the agent also sees. Candidate
scenarios are drawn from a fixed random seed, so results are deterministic.
"""

from __future__ import annotations

import math
import random

from .env import Config, sample_level

N_SCENARIOS = 4000
_LENGTH = Config().horizon + sum(Config().lead_times) + 2

_scenarios = None   # list of (inverse shape, cum. log shape, cum. shape, cum. shape^2)
_state_cache = {}   # history tuple -> per-scenario running sums
_post_cache = {}    # history tuple -> weighted scenarios
_quantile_cache = {}  # (history, weeks, probs) -> quantiles


def _build():
    global _scenarios
    rng = random.Random(20261113)
    out = []
    for _ in range(N_SCENARIOS):
        base, level, _cv = sample_level(rng, _LENGTH)
        shape = [l / base for l in level]
        inv = [1.0 / s for s in shape]
        cum_log, cum1, cum2 = [0.0], [0.0], [0.0]
        for s in shape:
            cum_log.append(cum_log[-1] + math.log(s))
            cum1.append(cum1[-1] + s)
            cum2.append(cum2[-1] + s * s)
        out.append((inv, cum_log, cum1, cum2))
    _scenarios = out


def _running_sums(history: tuple) -> tuple:
    """Per scenario: sum and sum of squares of demand divided by the
    scenario's shape. Dividing out the shape leaves the base level plus
    noise, so these two sums are all the fit needs."""
    if history in _state_cache:
        return _state_cache[history]
    if len(_state_cache) > 4000:
        _state_cache.clear()
    n = len(history)
    if n == 0:
        res = ([0.0] * N_SCENARIOS, [0.0] * N_SCENARIOS)
    else:
        sx, sxx = _running_sums(history[:-1])
        d, t = history[-1], n - 1
        xs = [d * sc[0][t] for sc in _scenarios]
        res = ([a + x for a, x in zip(sx, xs)],
               [a + x * x for a, x in zip(sxx, xs)])
    _state_cache[history] = res
    return res


def _posterior(history: tuple) -> list:
    """Scenarios that still fit the history, as (weight, base, sigma, index).
    For each scenario the base level and noise are fitted to the history
    (within the ranges stated in the rules) and the scenario is weighted by
    the likelihood of that fit."""
    if history in _post_cache:
        return _post_cache[history]
    if _scenarios is None:
        _build()
    if len(_post_cache) > 4000:
        _post_cache.clear()
    n = len(history)
    sx, sxx = _running_sums(history)
    fits = []
    for j in range(N_SCENARIOS):
        base = min(25.0, max(15.0, sx[j] / n))
        sse = max(0.0, sxx[j] - 2 * base * sx[j] + n * base * base)
        cv = min(0.30, max(0.15, math.sqrt(sse / n) / base))
        sigma = cv * base
        loglik = (-_scenarios[j][1][n] - n * math.log(sigma)
                  - sse / (2 * sigma * sigma))
        fits.append((loglik, base, sigma, j))
    top = max(f[0] for f in fits)
    kept = [(math.exp(f[0] - top), f[1], f[2], f[3]) for f in fits
            if f[0] > top - 9.0]
    total = sum(k[0] for k in kept)
    res = [(w / total, b, s, j) for w, b, s, j in kept]
    _post_cache[history] = res
    return res


def demand_quantiles(history, weeks: int, probs) -> list:
    """Quantiles of total demand over the next `weeks` weeks.

    The forecast is a weighted mixture of normal distributions, one per
    surviving scenario; each quantile is found by bisection on its CDF.
    """
    history = tuple(history)
    key = (history, weeks, tuple(probs))
    if key in _quantile_cache:
        return _quantile_cache[key]
    if len(_quantile_cache) > 20000:
        _quantile_cache.clear()
    n = len(history)
    parts = []
    for w, base, sigma, j in _posterior(history):
        _, _, cum1, cum2 = _scenarios[j]
        mean = base * (cum1[n + weeks] - cum1[n])
        sd = sigma * math.sqrt(cum2[n + weeks] - cum2[n])
        parts.append((w, mean, sd * math.sqrt(2.0)))
    lo0 = min(m - 4 * s for _, m, s in parts)
    hi0 = max(m + 4 * s for _, m, s in parts)
    out = []
    for p in probs:
        lo, hi = lo0, hi0
        for _ in range(24):
            mid = (lo + hi) / 2
            cdf = sum(w * 0.5 * (1 + math.erf((mid - m) / s)) for w, m, s in parts)
            if cdf < p:
                lo = mid
            else:
                hi = mid
        out.append(max(0.0, (lo + hi) / 2))
    _quantile_cache[key] = out
    return out
