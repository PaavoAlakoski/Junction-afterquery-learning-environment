#!/bin/bash
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
open("/logs/verifier/reward.txt", "w").write(f"{reward:.4f}\n")
open("/logs/verifier/result.json", "w").write(json.dumps(result))
PY
