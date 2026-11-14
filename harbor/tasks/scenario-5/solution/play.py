"""Plays a scripted policy through the MCP server, as an agent would."""

import json
import sys
import urllib.request

sys.path.insert(0, "/solution")
from supplychain import policies  # noqa: E402

URL = "http://supply-chain:8000/mcp"
STOP_EARLY_AFTER = 10
_id = 0


def rpc(method, params=None):
    global _id
    _id += 1
    body = json.dumps({"jsonrpc": "2.0", "id": _id, "method": method,
                       "params": params or {}}).encode()
    req = urllib.request.Request(URL, body, {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream"})
    with urllib.request.urlopen(req, timeout=60) as f:
        return json.loads(f.read())["result"]


def tool(name, **args):
    return json.loads(rpc("tools/call", {"name": name, "arguments": args})["content"][0]["text"])


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "reference"
    rpc("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                       "clientInfo": {"name": "reference-run", "version": "1"}})
    if mode == "nothing":
        print("placing no orders")
        return
    out = {"done": False, "state": tool("get_state")}
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
