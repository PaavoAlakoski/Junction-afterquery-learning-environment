"""Builds replay.html: a week-by-week comparison of any saved agent run with
the reference run on the same scenario.

    python replay.py

It reads every results/agent-*.json file, replays the saved orders through
the simulator, and writes one self-contained page (no network needed) showing
demand, stock in the system and customer backlog for the chosen run next to
the reference. The naive rule and the trained small policy are included too.
"""

from __future__ import annotations

import glob
import json
import pathlib

from supplychain import SupplyChainEnv, learned, policies

SEEDS = [1, 2, 3, 4, 5]


def trace(seed: int, policy=None, orders=None) -> dict:
    """Play one episode from a policy or from a saved list of orders."""
    env = SupplyChainEnv(seed)
    obs = env.observe()
    out = {"demand": [], "stock": [], "backlog": [], "retail": [],
           "hold": 0.0, "short": 0.0}
    i = 0
    while True:
        out["demand"].append(env.demand_history[-1])
        out["stock"].append(env.system_stock())
        out["backlog"].append(env.backlog)
        out["retail"].append(env.on_hand[0])
        out["hold"] += sum(h * s for h, s in zip(env.cfg.holding_cost, env.on_hand))
        out["short"] += env.cfg.backlog_cost * env.backlog
        action = orders[i] if orders is not None else policy(obs)
        i += 1
        obs = env.step(action).observation
        if env.done:
            break
    out["cost"] = round(env.total_cost)
    out["hold"], out["short"] = round(out["hold"]), round(out["short"])
    out["handover"] = round(env.handover_penalty)
    return out


def collect() -> dict:
    runs = {}   # label -> {seed: trace}
    for path in sorted(glob.glob("results/agent-*.json")):
        data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        model = pathlib.Path(path).stem.split("agent-")[1].rsplit("-", 2)[0]
        for name, res in data["results"].items():
            if "transcripts" not in res:
                continue
            label = f"{model}, {name.replace('agent ', '')}"
            runs[label] = {
                s: trace(s, orders=[w["orders"] for w in res["transcripts"][str(s)]])
                for s in SEEDS if str(s) in res["transcripts"]}
    trained = learned.load_trained()
    if trained:
        runs["small policy, trained"] = {s: trace(s, trained) for s in SEEDS}
    runs["naive pass-through"] = {s: trace(s, policies.pass_through) for s in SEEDS}
    reference = {s: trace(s, policies.reference) for s in SEEDS}
    return {"seeds": SEEDS, "reference": reference, "runs": runs}


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Supply chain replay</title>
<style>
:root { --bg:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --grid:#e4e3df;
        --ref:#2a78d6; --agent:#eb6834; --demand:#52514e; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#0d0d0d; --surface:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7; --grid:#383835;
          --ref:#3987e5; --agent:#d95926; --demand:#c3c2b7; } }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--ink);
       font:15px/1.45 "Segoe UI", system-ui, Arial, sans-serif; }
main { max-width:1000px; margin:0 auto; padding:20px 16px 40px; }
h1 { font-size:22px; margin:0 0 4px; }
p.lead { color:var(--ink2); margin:0 0 14px; }
.controls { display:flex; gap:14px; flex-wrap:wrap; margin-bottom:14px; }
label { font-size:13px; color:var(--ink2); display:flex; flex-direction:column; gap:3px; }
select { font:inherit; padding:5px 8px; background:var(--surface); color:var(--ink);
         border:1px solid var(--grid); border-radius:4px; }
.tiles { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:10px; margin-bottom:14px; }
.tile { background:var(--surface); border:1px solid var(--grid); border-radius:6px; padding:10px 12px; }
.tile .k { font-size:12px; color:var(--ink2); }
.tile .v { font-size:22px; font-weight:600; font-variant-numeric:tabular-nums; }
.tile .s { font-size:12px; color:var(--ink2); }
.card { background:var(--surface); border:1px solid var(--grid); border-radius:6px;
        padding:12px 12px 6px; margin-bottom:12px; position:relative; }
.card h2 { font-size:15px; margin:0 0 2px; }
.card .note { font-size:12.5px; color:var(--ink2); margin:0 0 4px; }
.legend { display:flex; gap:16px; font-size:12.5px; color:var(--ink2); margin:2px 0 4px; }
.sw { display:inline-block; width:14px; height:3px; vertical-align:middle; margin-right:5px; border-radius:2px; }
svg { width:100%; height:auto; display:block; }
svg text { fill:var(--ink2); font-size:11px; }
.tip { position:absolute; pointer-events:none; background:var(--surface); border:1px solid var(--grid);
       border-radius:4px; padding:5px 8px; font-size:12px; box-shadow:0 2px 6px rgba(0,0,0,.15);
       display:none; white-space:nowrap; z-index:2; }
table { border-collapse:collapse; width:100%; font-size:13px; font-variant-numeric:tabular-nums; }
th,td { text-align:right; padding:4px 8px; border-bottom:1px solid var(--grid); }
th:first-child, td:first-child { text-align:left; }
</style></head><body><main>
<h1>Supply chain replay</h1>
<p class="lead">One scenario, two players: the chosen run and the reference run, facing the same customer demand.</p>
<div class="controls">
  <label>Scenario <select id="seed"></select></label>
  <label>Run to compare <select id="run"></select></label>
</div>
<div class="tiles" id="tiles"></div>
<div class="card"><h2>Customer demand per week</h2>
  <p class="note">What both players face. Neither sees future weeks.</p><div id="c0"></div></div>
<div class="card"><h2>Stock in the system</h2>
  <p class="note">Units on hand and in transit across all four stages, minus customer backlog.</p>
  <div class="legend" id="l1"></div><div id="c1"></div></div>
<div class="card"><h2>Customer backlog</h2>
  <p class="note">Unfilled customer demand. Each unit costs 10 per week.</p>
  <div class="legend" id="l2"></div><div id="c2"></div></div>
<div class="card"><h2>Cost breakdown</h2><div id="tbl"></div></div>
</main>
<script>
const DATA = __DATA__;
const $ = id => document.getElementById(id);
const fmt = n => Math.round(n).toLocaleString("en-US");
const seedSel = $("seed"), runSel = $("run");
DATA.seeds.forEach(s => seedSel.add(new Option("Scenario " + s, s)));
Object.keys(DATA.runs).forEach(r => runSel.add(new Option(r, r)));
seedSel.value = 3;
const strongest = Object.keys(DATA.runs).find(r => r.includes("sonnet")); if (strongest) runSel.value = strongest;

function chart(el, series, yLabel) {
  const W = 960, H = 210, L = 44, R = 12, T = 10, B = 26, n = series[0].data.length;
  const raw = Math.max(4, ...series.flatMap(s => s.data)), min = 0;
  const unit = [1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500].find(u => u * 4 >= raw) || 1000, max = unit * 4;
  const x = i => L + (W - L - R) * i / (n - 1), y = v => T + (H - T - B) * (1 - (v - min) / (max - min));
  let g = "";
  for (let k = 0; k <= 4; k++) { const v = min + (max - min) * k / 4;
    g += `<line x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}" stroke="var(--grid)"/>` +
         `<text x="${L - 6}" y="${y(v) + 4}" text-anchor="end">${fmt(v)}</text>`; }
  for (let w = 1; w <= n; w++) if (w === 1 || w % 4 === 0)
    g += `<text x="${x(w - 1)}" y="${H - 8}" text-anchor="middle">${w}</text>`;
  g += `<text x="${W - R}" y="${H - 8}" text-anchor="end" dx="0" dy="0"></text>`;
  series.forEach(s => { g += `<path fill="none" stroke="${s.color}" stroke-width="2" stroke-linejoin="round" d="M` +
    s.data.map((v, i) => x(i).toFixed(1) + " " + y(v).toFixed(1)).join(" L") + `"/>`; });
  g += `<line id="cross" y1="${T}" y2="${H - B}" stroke="var(--ink2)" stroke-dasharray="3 3" visibility="hidden"/>`;
  el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${yLabel}">${g}</svg><div class="tip"></div>`;
  const svg = el.querySelector("svg"), tip = el.querySelector(".tip"), cross = el.querySelector("#cross");
  svg.onmousemove = e => { const r = svg.getBoundingClientRect();
    const i = Math.max(0, Math.min(n - 1, Math.round(((e.clientX - r.left) / r.width * W - L) / (W - L - R) * (n - 1))));
    cross.setAttribute("x1", x(i)); cross.setAttribute("x2", x(i)); cross.setAttribute("visibility", "visible");
    tip.innerHTML = `<b>Week ${i + 1}</b><br>` + series.map(s =>
      `<span class="sw" style="background:${s.color}"></span>${s.name}: ${fmt(s.data[i])}`).join("<br>");
    tip.style.display = "block";
    const cardR = el.parentElement.getBoundingClientRect();
    let left = e.clientX - cardR.left + 14; if (left > cardR.width - 190) left -= 200;
    tip.style.left = left + "px"; tip.style.top = (e.clientY - cardR.top - 10) + "px"; };
  svg.onmouseleave = () => { tip.style.display = "none"; cross.setAttribute("visibility", "hidden"); };
}
function legend(el, name) {
  el.innerHTML = `<span><span class="sw" style="background:var(--ref)"></span>Reference</span>` +
                 `<span><span class="sw" style="background:var(--agent)"></span>${name}</span>`;
}
function draw() {
  const s = seedSel.value, name = runSel.value, ref = DATA.reference[s], run = DATA.runs[name][s];
  const tile = (k, v, sub) => `<div class="tile"><div class="k">${k}</div><div class="v">${v}</div><div class="s">${sub}</div></div>`;
  $("tiles").innerHTML =
    tile("Reference cost", fmt(ref.cost), "lower is better") +
    tile(name + " cost", fmt(run.cost), (run.cost > ref.cost ? "+" : "") + fmt(run.cost - ref.cost) + " vs reference") +
    tile("Worst backlog, reference", fmt(Math.max(...ref.backlog)), "units") +
    tile("Worst backlog, " + name, fmt(Math.max(...run.backlog)), "units");
  chart($("c0"), [{name: "Demand", color: "var(--demand)", data: ref.demand}], "Customer demand per week");
  legend($("l1"), name); legend($("l2"), name);
  const pair = k => [{name: "Reference", color: "var(--ref)", data: ref[k]}, {name: name, color: "var(--agent)", data: run[k]}];
  chart($("c1"), pair("stock"), "Stock in the system per week");
  chart($("c2"), pair("backlog"), "Customer backlog per week");
  const row = (k, a, b) => `<tr><td>${k}</td><td>${fmt(a)}</td><td>${fmt(b)}</td></tr>`;
  $("tbl").innerHTML = `<table><tr><th>Cost</th><th>Reference</th><th>${name}</th></tr>` +
    row("Holding stock", ref.hold, run.hold) + row("Customer backlog", ref.short, run.short) +
    row("Handover penalty", ref.handover, run.handover) + row("Total", ref.cost, run.cost) + `</table>`;
}
seedSel.onchange = runSel.onchange = draw; draw();
</script></body></html>
"""


def main():
    data = collect()
    out = pathlib.Path("replay.html")
    out.write_text(PAGE.replace("__DATA__", json.dumps(data)), encoding="utf-8")
    print(f"Wrote {out} ({len(data['runs'])} runs x {len(SEEDS)} scenarios)")


if __name__ == "__main__":
    main()
