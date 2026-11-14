"""Generates the Harbor tasks in harbor/tasks/, one per scenario.

    python harbor/build_tasks.py

Each task is self-contained: Harbor builds its containers from the task's own
environment/ folder, so this script copies the simulator into every task. Run
it again after changing anything in supplychain/ or mcp_server.py.

Layout of one task (Harbor task format):
    task.toml                      config; declares the MCP server
    instruction.md                 what the agent is told
    environment/Dockerfile         the agent's container ("main")
    environment/docker-compose.yaml   adds the MCP server as a sidecar
    environment/mcp-server/        the simulator; the agent cannot read it
    solution/solve.sh              reference run (Harbor's oracle agent)
    tests/test.sh                  reads the score from the sidecar
"""

from __future__ import annotations

import pathlib
import shutil

ROOT = pathlib.Path(__file__).resolve().parent.parent
TASKS = ROOT / "harbor" / "tasks"
SEEDS = [1, 2, 3, 4, 5]
STOP_EARLY_AFTER = 10

TASK_TOML = '''schema_version = "1.4"

[task]
name = "rome-48h/delayed-supply-chain-{seed}"
description = "Run a four-stage supply chain for 52 weeks at the lowest total cost (scenario {seed})."
keywords = ["planning", "inventory", "delayed-effects"]

[metadata]
category = "planning"

[agent]
timeout_sec = 3600.0

[verifier]
timeout_sec = 120.0

[environment]
cpus = 1
memory_mb = 2048

[[environment.mcp_servers]]
name = "supply-chain"
transport = "streamable-http"
url = "http://supply-chain:8000/mcp"
'''

INSTRUCTION = '''# Run the supply chain

You manage a four-stage supply chain (factory, distributor, wholesaler,
retailer) for 52 weeks. Your goal is the lowest total cost.

Everything happens through the `supply-chain` MCP server:

- `get_rules` returns the full rules: how a week works, every cost, how
  customer demand behaves, and the end-of-episode handover rule. Read it first.
- `get_state` returns the current state.
- `place_orders` places this week's four order quantities and moves on one
  week. It returns the new state.
- `stop` ends the episode early. Every remaining week is then played with zero
  orders, which is very costly.

Play all 52 weeks. The episode is finished when `place_orders` returns
`"done": true` with your final score. There is one episode and it cannot be
restarted.

Your result is the score reported by the server (0 to 100, higher is better).
Nothing needs to be written to disk.
'''

MAIN_DOCKERFILE = '''FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates git \\
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
'''

COMPOSE = '''services:
  main:
    depends_on:
      supply-chain:
        condition: service_healthy
  supply-chain:
    build:
      context: ./mcp-server
    expose:
      - "8000"
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"]
      interval: 2s
      timeout: 5s
      retries: 30
'''

SERVER_DOCKERFILE = '''FROM python:3.12-slim
WORKDIR /srv
COPY . /srv
CMD ["python", "mcp_server.py", "--seed", "{seed}", "--transport", "http", "--host", "0.0.0.0", "--port", "8000", "--result-file", "/srv/result.json"]
'''

SOLVE_SH = '''#!/bin/bash
# Reference run. Plays the reference policy through the MCP server.
#   SOLUTION=reference   (default) the reference run
#   SOLUTION=stop_early  the reference for {stop} weeks, then stop
#   SOLUTION=nothing     place no orders at all
set -euo pipefail
python3 /solution/play.py "${{SOLUTION:-reference}}"
'''

PLAY_PY = '''"""Plays a scripted policy through the MCP server, as an agent would."""

import json
import sys
import urllib.request

sys.path.insert(0, "/solution")
from supplychain import policies  # noqa: E402

URL = "http://supply-chain:8000/mcp"
STOP_EARLY_AFTER = {stop}
_id = 0


def rpc(method, params=None):
    global _id
    _id += 1
    body = json.dumps({{"jsonrpc": "2.0", "id": _id, "method": method,
                       "params": params or {{}}}}).encode()
    req = urllib.request.Request(URL, body, {{
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream"}})
    with urllib.request.urlopen(req, timeout=60) as f:
        return json.loads(f.read())["result"]


def tool(name, **args):
    return json.loads(rpc("tools/call", {{"name": name, "arguments": args}})["content"][0]["text"])


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "reference"
    rpc("initialize", {{"protocolVersion": "2025-06-18", "capabilities": {{}},
                       "clientInfo": {{"name": "reference-run", "version": "1"}}}})
    if mode == "nothing":
        print("placing no orders")
        return
    out = {{"done": False, "state": tool("get_state")}}
    while not out["done"]:
        state = out["state"]
        if mode == "stop_early" and state["week"] > STOP_EARLY_AFTER:
            out = tool("stop")
            break
        q = policies.reference(state)
        out = tool("place_orders", retailer=q[0], wholesaler=q[1],
                   distributor=q[2], factory=q[3])
    print(json.dumps(out))


if __name__ == "__main__":
    main()
'''

TEST_SH = '''#!/bin/bash
# Reads the score from the MCP server and writes the reward (score / 100).
# An episode that was never played or was stopped early is scored as zero
# orders for the remaining weeks.
set -uo pipefail
mkdir -p /logs/verifier
python3 - <<'PY'
import json, urllib.request
try:
    with urllib.request.urlopen("http://supply-chain:8000/result", timeout=30) as f:
        result = json.loads(f.read())
    reward = max(0.0, min(1.0, result["score"] / 100.0))
except Exception as e:           # no result means no credit
    result, reward = {"error": repr(e)}, 0.0
print(json.dumps(result))
open("/logs/verifier/reward.txt", "w").write(f"{reward:.4f}\\n")
open("/logs/verifier/result.json", "w").write(json.dumps(result))
PY
'''


def write(path: pathlib.Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def copy_package(dest: pathlib.Path):
    target = dest / "supplychain"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(ROOT / "supplychain", target,
                    ignore=shutil.ignore_patterns("__pycache__", "claude_agent.py",
                                                  "llm_agent.py"))


def main():
    for seed in SEEDS:
        task = TASKS / f"scenario-{seed}"
        if task.exists():
            shutil.rmtree(task)
        write(task / "task.toml", TASK_TOML.format(seed=seed))
        write(task / "instruction.md", INSTRUCTION)
        write(task / "environment" / "Dockerfile", MAIN_DOCKERFILE)
        write(task / "environment" / "docker-compose.yaml", COMPOSE)
        server = task / "environment" / "mcp-server"
        write(server / "Dockerfile", SERVER_DOCKERFILE.format(seed=seed))
        shutil.copy(ROOT / "mcp_server.py", server / "mcp_server.py")
        copy_package(server)
        write(task / "solution" / "solve.sh", SOLVE_SH.format(stop=STOP_EARLY_AFTER))
        write(task / "solution" / "play.py", PLAY_PY.format(stop=STOP_EARLY_AFTER))
        copy_package(task / "solution")
        write(task / "tests" / "test.sh", TEST_SH)
    print(f"Wrote {len(SEEDS)} tasks to {TASKS}")


if __name__ == "__main__":
    main()
