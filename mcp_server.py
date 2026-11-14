"""MCP server for the Delayed Supply Chain environment.

    python mcp_server.py --seed 1                     stdio (default)
    python mcp_server.py --seed 1 --transport http    HTTP at http://127.0.0.1:8000/mcp
                                                      (also GET /result and /health)

One server process plays exactly one episode of one scenario. The agent
cannot choose the scenario or start again: demand is the same every time a
scenario is played, so a restart would let it play with knowledge of the
future. To evaluate a suite, start one server per seed.

Tools offered to the agent:
    get_rules()                      the rules of the task
    get_state()                      the current week's state
    place_orders(retailer, wholesaler, distributor, factory)
                                     place this week's four orders; moves on one week
    stop()                           give up; remaining weeks get zero orders

The score is not something the agent reports. After every week the server
writes the result to --result-file (default results/mcp-seed-<seed>.json):
the score the episode would get if the agent stopped now. An agent that never
calls a tool, or stops early, is therefore scored as placing zero orders for
the rest of the episode.

Uses only the Python standard library. It implements the parts of the Model
Context Protocol a tool server needs: initialize, ping, tools/list and
tools/call, over stdio (one JSON-RPC message per line) or HTTP POST.
"""

from __future__ import annotations

import argparse
import copy
import json
import pathlib
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from supplychain import SupplyChainEnv, rules_text
from supplychain.scoring import anchors, score

PROTOCOL_VERSIONS = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")
DEFAULT_PROTOCOL = "2025-06-18"
SERVER_INFO = {"name": "delayed-supply-chain", "version": "1.0.0"}
INSTRUCTIONS = (
    "You run a four-stage supply chain for 52 weeks and want the lowest total "
    "cost. Call get_rules first. Then, every week, read the state and call "
    "place_orders with one order quantity per stage. The episode ends after "
    "the week-52 orders. There is one episode; it cannot be restarted.")

_ORDER = {"type": "integer", "minimum": 0, "maximum": 200}
TOOLS = [
    {"name": "get_rules",
     "description": "The full rules of the task: how a week works, all costs, "
                    "how customer demand behaves, and the end-of-episode "
                    "handover rule.",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "get_state",
     "description": "The current state: week, stock on hand and in transit at "
                    "every stage, backorders, customer backlog, the full "
                    "demand and order history, and cost so far. Reading it "
                    "does not advance time.",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "place_orders",
     "description": "Place this week's order for each stage and move on one "
                    "week. Each quantity is a whole number from 0 to 200; the "
                    "factory's is a production order capped at 80. Returns "
                    "the cost charged for the new week and the new state. "
                    "After the week-52 orders it returns the final score.",
     "inputSchema": {"type": "object",
                     "properties": {
                         "retailer": {**_ORDER, "description": "Units the retailer orders from the wholesaler."},
                         "wholesaler": {**_ORDER, "description": "Units the wholesaler orders from the distributor."},
                         "distributor": {**_ORDER, "description": "Units the distributor orders from the factory."},
                         "factory": {**_ORDER, "description": "Units the factory starts producing (at most 80)."}},
                     "required": ["retailer", "wholesaler", "distributor", "factory"]}},
    {"name": "stop",
     "description": "End the episode now. Every remaining week is played with "
                    "zero orders, which is very costly. This cannot be undone.",
     "inputSchema": {"type": "object", "properties": {}}},
]


class Episode:
    """One scenario, played once. All tool calls go through `call`."""

    def __init__(self, seed: int, result_file: pathlib.Path):
        self.seed = seed
        self.env = SupplyChainEnv(seed)
        self.naive, self.floor = anchors(seed)
        self.result_file = result_file
        self.lock = threading.Lock()
        self.write_result()

    def result(self) -> dict:
        """The result if no further orders are placed."""
        env = self.env if self.env.done else copy.deepcopy(self.env)
        weeks_played = len(self.env.order_history)
        env.finish()
        return {
            "seed": self.seed,
            "finished": self.env.done,
            "weeks_played": weeks_played,
            "score": round(score(env.total_cost, self.naive, self.floor), 1),
            "total_cost": round(env.total_cost, 1),
            "handover_penalty": round(env.handover_penalty, 1),
            "invalid_actions": self.env.invalid_actions,
            "naive_cost": round(self.naive, 1),
            "floor_cost": round(self.floor, 1),
        }

    def write_result(self):
        self.result_file.parent.mkdir(parents=True, exist_ok=True)
        self.result_file.write_text(json.dumps(self.result(), indent=2))

    def final(self) -> dict:
        r = self.result()
        return {"done": True, "score": r["score"], "total_cost": r["total_cost"],
                "handover_penalty": r["handover_penalty"]}

    def call(self, name: str, args: dict):
        with self.lock:
            if name == "get_rules":
                return rules_text(self.env.cfg)
            if name == "get_state":
                return self.env.observe()
            if name == "place_orders":
                if self.env.done:
                    return {"error": "The episode is finished.", **self.final()}
                # Missing or malformed quantities count as an order of 0,
                # exactly as the rules say.
                res = self.env.step([args.get(k) for k in
                                     ("retailer", "wholesaler", "distributor", "factory")])
                self.write_result()
                if res.done:
                    return self.final()
                # The reply leaves out the order history (the agent's own
                # past orders) to keep it short; get_state has everything.
                state = {k: v for k, v in res.observation.items()
                         if k != "order_history"}
                return {"done": False, "cost_this_week": res.cost, "state": state}
            if name == "stop":
                if not self.env.done:
                    self.env.finish()
                    self.write_result()
                return self.final()
            raise KeyError(name)


def handle(episode: Episode, msg):
    """Answer one JSON-RPC message. Returns None for notifications."""
    if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0":
        return {"jsonrpc": "2.0", "id": None,
                "error": {"code": -32600, "message": "Invalid request"}}
    if "id" not in msg or "method" not in msg:
        return None                       # a notification, or a response to us
    method, params = msg["method"], msg.get("params") or {}

    def ok(result):
        return {"jsonrpc": "2.0", "id": msg["id"], "result": result}

    def err(code, message):
        return {"jsonrpc": "2.0", "id": msg["id"],
                "error": {"code": code, "message": message}}

    if method == "initialize":
        wanted = params.get("protocolVersion")
        return ok({
            "protocolVersion": wanted if wanted in PROTOCOL_VERSIONS else DEFAULT_PROTOCOL,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": SERVER_INFO,
            "instructions": INSTRUCTIONS})
    if method == "ping":
        return ok({})
    if method == "tools/list":
        return ok({"tools": TOOLS})
    if method == "tools/call":
        name, args = params.get("name"), params.get("arguments") or {}
        if name not in {t["name"] for t in TOOLS}:
            return err(-32602, f"Unknown tool: {name}")
        if not isinstance(args, dict):
            return err(-32602, "Tool arguments must be an object")
        out = episode.call(name, args)
        text = out if isinstance(out, str) else json.dumps(out)
        return ok({"content": [{"type": "text", "text": text}], "isError": False})
    return err(-32601, f"Method not found: {method}")


def handle_payload(episode: Episode, payload):
    """A payload is one message or a batch of them."""
    if isinstance(payload, list):
        replies = [r for r in (handle(episode, m) for m in payload) if r is not None]
        return replies or None
    return handle(episode, payload)


def serve_stdio(episode: Episode):
    out = sys.stdout.buffer
    for raw in sys.stdin.buffer:
        raw = raw.strip()
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except ValueError:
            reply = {"jsonrpc": "2.0", "id": None,
                     "error": {"code": -32700, "message": "Parse error"}}
        else:
            reply = handle_payload(episode, payload)
        if reply is not None:
            out.write(json.dumps(reply).encode("utf-8") + b"\n")
            out.flush()


def serve_http(episode: Episode, host: str, port: int):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: bytes = b""):
            self.send_response(status)
            if body:
                self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if self.path.rstrip("/") != "/mcp":
                return self._send(404)
            try:
                length = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(length))
            except ValueError:
                return self._send(400, json.dumps(
                    {"jsonrpc": "2.0", "id": None,
                     "error": {"code": -32700, "message": "Parse error"}}).encode())
            reply = handle_payload(episode, payload)
            if reply is None:
                return self._send(202)
            self._send(200, json.dumps(reply).encode("utf-8"))

        def do_GET(self):
            path = self.path.rstrip("/")
            if path == "/health":
                return self._send(200, b'{"ok": true}')
            if path == "/result":
                # For the verifier. Leaves out the scoring anchors, which
                # depend on future demand.
                with episode.lock:
                    r = episode.result()
                public = {k: r[k] for k in ("seed", "finished", "weeks_played", "score",
                                            "total_cost", "handover_penalty", "invalid_actions")}
                return self._send(200, json.dumps(public).encode())
            self._send(405)            # no server-initiated stream is offered

        def do_DELETE(self):
            self._send(405)

        def log_message(self, *args):
            pass

    print(f"MCP server on http://{host}:{port}/mcp (seed {episode.seed})",
          file=sys.stderr, flush=True)
    ThreadingHTTPServer((host, port), Handler).serve_forever()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--seed", type=int, default=1, help="scenario to play")
    ap.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--result-file", help="where the score is written")
    args = ap.parse_args()
    episode = Episode(args.seed, pathlib.Path(
        args.result_file or f"results/mcp-seed-{args.seed}.json"))
    if args.transport == "stdio":
        serve_stdio(episode)
    else:
        serve_http(episode, args.host, args.port)


if __name__ == "__main__":
    main()
