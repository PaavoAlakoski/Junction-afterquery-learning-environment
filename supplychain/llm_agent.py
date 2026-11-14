"""A minimal LLM agent, for measuring the gap before the official harness is
wired in. It talks to any OpenAI-compatible chat-completions endpoint.

Set three environment variables:
    LLM_BASE_URL   e.g. https://your-provider.example/v1
    LLM_API_KEY
    LLM_MODEL

Each week the model receives the rules, the full observation and the notes it
wrote to itself last week, and must reply with JSON:
    {"orders": [retailer, wholesaler, distributor, factory], "notes": "..."}
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request

from .env import N, rules_text

SYSTEM = rules_text() + """
Each turn you get the current state as JSON. Think it through, then end your
reply with one JSON object on its own line:
{"orders": [q0, q1, q2, q3], "notes": "anything you want to remember next week"}
"""


def _chat(messages: list) -> str:
    base = os.environ["LLM_BASE_URL"].rstrip("/")
    body = json.dumps({"model": os.environ["LLM_MODEL"],
                       "messages": messages}).encode()
    req = urllib.request.Request(
        base + "/chat/completions", data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + os.environ["LLM_API_KEY"]})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.load(r)["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 529) or attempt == 4:
                raise
        except urllib.error.URLError:
            if attempt == 4:
                raise
        time.sleep(2 ** attempt * 2)
    raise RuntimeError("unreachable")


def _parse(text: str):
    """Take the last JSON object in the reply that has an "orders" list."""
    for m in reversed(re.findall(r"\{[^{}]*\}", text or "", flags=re.S)):
        try:
            obj = json.loads(m)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("orders"), list):
            return obj
    return None


class LLMAgent:
    """Callable policy. Keeps the model's own notes between weeks and a full
    transcript for failure analysis."""

    def __init__(self):
        for var in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"):
            if not os.environ.get(var):
                raise SystemExit(f"Set {var} to run the LLM agent "
                                 "(see supplychain/llm_agent.py).")
        self.notes = ""
        self.transcript = []

    def __call__(self, obs: dict) -> list:
        user = ("State:\n" + json.dumps(obs) +
                "\n\nYour notes from last week:\n" + (self.notes or "(none)"))
        reply = _chat([{"role": "system", "content": SYSTEM},
                       {"role": "user", "content": user}])
        parsed = _parse(reply)
        orders = parsed["orders"] if parsed else [None] * N
        if parsed and isinstance(parsed.get("notes"), str):
            self.notes = parsed["notes"][:4000]
        self.transcript.append({"week": obs["week"], "reply": reply,
                                "orders": orders})
        return orders
