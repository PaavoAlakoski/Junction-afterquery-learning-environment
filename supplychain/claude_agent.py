"""Claude agent, via the official Anthropic SDK (`pip install anthropic`).

Credentials are read from the environment (ANTHROPIC_API_KEY) by the SDK.
Optional settings:
    CLAUDE_MODEL    default claude-haiku-5-5 (cheapest; use claude-sonnet-5-5
                    or claude-opus-5-5 for a stronger agent)
    CLAUDE_EFFORT   low | medium | high | xhigh | max   (default medium)
    CLAUDE_MAX_USD  spending cap for one evaluate.py invocation (default 2)

Each week the model receives the rules, the full observation and the notes it
wrote to itself last week, and replies with JSON:
    {"orders": [retailer, wholesaler, distributor, factory], "notes": "..."}

No fallback model is configured, on purpose: a score should belong to exactly
one model. A refusal or API failure aborts the evaluation instead of being
counted as the agent's mistake.
"""

from __future__ import annotations

import json
import os

import anthropic

from .env import N
from .llm_agent import SYSTEM, _parse

DEFAULT_MODEL = "claude-haiku-5-5"

# USD per million tokens (input, output), used only for the spending cap.
PRICES = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
}

_client = None
usage_totals = {"input_tokens": 0, "output_tokens": 0, "calls": 0, "usd": 0.0}


def _max_usd() -> float:
    return float(os.environ.get("CLAUDE_MAX_USD", "2"))


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(max_retries=6)
    return _client


class ClaudeAgent:
    """Callable policy. Keeps the model's own notes between weeks and a full
    transcript for failure analysis."""

    def __init__(self):
        self.model = os.environ.get("CLAUDE_MODEL", DEFAULT_MODEL)
        self.effort = os.environ.get("CLAUDE_EFFORT", "medium")
        if self.model not in PRICES:
            raise SystemExit(f"No price listed for {self.model}; add it to "
                             "PRICES in supplychain/claude_agent.py so the "
                             "spending cap can work.")
        self.notes = ""
        self.transcript = []

    def __call__(self, obs: dict) -> list:
        if usage_totals["usd"] >= _max_usd():
            raise SystemExit(
                f"Spending cap reached: ${usage_totals['usd']:.2f} of "
                f"${_max_usd():.2f} (CLAUDE_MAX_USD). Evaluation aborted.")
        user = ("State:\n" + json.dumps(obs) +
                "\n\nYour notes from last week:\n" + (self.notes or "(none)"))
        try:
            response = _get_client().messages.create(
                model=self.model,
                max_tokens=16000,
                thinking={"type": "adaptive"},
                output_config={"effort": self.effort},
                system=SYSTEM,
                messages=[{"role": "user", "content": user}],
            )
        except anthropic.AuthenticationError:
            raise SystemExit("Anthropic API key missing or invalid. "
                             "Set ANTHROPIC_API_KEY and run again.")
        except anthropic.APIError as e:
            raise SystemExit(f"Anthropic API error, evaluation aborted: {e}")
        except TypeError as e:  # the SDK raises this when no credentials are set
            raise SystemExit(f"Set ANTHROPIC_API_KEY and run again. ({e})")
        if response.stop_reason == "refusal":
            raise SystemExit("The model refused a request; evaluation aborted.")

        usage_totals["input_tokens"] += response.usage.input_tokens
        usage_totals["output_tokens"] += response.usage.output_tokens
        usage_totals["calls"] += 1
        price_in, price_out = PRICES[self.model]
        usage_totals["usd"] += (response.usage.input_tokens * price_in +
                                response.usage.output_tokens * price_out) / 1e6

        reply = "".join(b.text for b in response.content if b.type == "text")
        parsed = _parse(reply)
        orders = parsed["orders"] if parsed else [None] * N
        if parsed and isinstance(parsed.get("notes"), str):
            self.notes = parsed["notes"][:4000]
        self.transcript.append({"week": obs["week"], "reply": reply,
                                "orders": orders})
        return orders
