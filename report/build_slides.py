"""Builds report/slides.html from the saved results. Run from the project root:

    python report/build_slides.py

Then print slides.html to PDF from a browser (landscape, no margins).
"""

import json
import pathlib
import sys

sys.path.insert(0, ".")
from replay import trace                      # noqa: E402
from supplychain import policies              # noqa: E402

REF, AGENT, SMALL, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#9a9993"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def line_chart(series, width=560, height=230, ymax=None, xlabel="Week",
               xs=None, labels=True, label_room=96):
    """series: list of (name, colour, values). Returns an SVG string."""
    L, R, T, B = 40, label_room if labels else 12, 10, 34
    n = len(series[0][2])
    xs = xs or list(range(1, n + 1))
    top = ymax or max(max(v) for _, _, v in series)
    unit = next(u for u in (1, 2, 5, 10, 20, 25, 50, 100, 125, 250, 500) if u * 4 >= top)
    top = unit * 4
    x = lambda i: L + (width - L - R) * i / (n - 1)
    y = lambda v: T + (height - T - B) * (1 - v / top)
    out = [f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">']
    for k in range(5):
        v = unit * k
        out.append(f'<line x1="{L}" x2="{width - R}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="{GRID}"/>'
                   f'<text x="{L - 6}" y="{y(v) + 4:.1f}" text-anchor="end" font-size="11" fill="{INK2}">{v}</text>')
    step = max(1, n // 7)
    for i in range(0, n, step):
        out.append(f'<text x="{x(i):.1f}" y="{height - 18}" text-anchor="middle" font-size="11" fill="{INK2}">{xs[i]}</text>')
    out.append(f'<text x="{(L + width - R) / 2}" y="{height - 3}" text-anchor="middle" font-size="11" fill="{INK2}">{xlabel}</text>')
    ends = []
    for name, colour, vals in series:
        pts = " L".join(f"{x(i):.1f} {y(v):.1f}" for i, v in enumerate(vals))
        out.append(f'<path d="M{pts}" fill="none" stroke="{colour}" stroke-width="2.2" stroke-linejoin="round"/>')
        ends.append([y(vals[-1]), name, colour])
    if labels:
        ends.sort()
        for i in range(1, len(ends)):          # keep end labels from colliding
            ends[i][0] = max(ends[i][0], ends[i - 1][0] + 14)
        for yy, name, colour in ends:
            out.append(f'<circle cx="{width - R + 8}" cy="{yy - 4:.1f}" r="4" fill="{colour}"/>'
                       f'<text x="{width - R + 16}" y="{yy:.1f}" font-size="11.5" fill="{INK}">{name}</text>')
    out.append("</svg>")
    return "".join(out)


def bars(rows):
    """rows: (label, value, colour). Horizontal bars on a 0-100 scale."""
    out = ['<div class="bars">']
    for label, value, colour in rows:
        out.append(f'<div class="brow"><div class="blab">{label}</div><div class="btrack">'
                   f'<div class="bbar" style="width:{max(value, 0.3)}%;background:{colour}"></div></div>'
                   f'<div class="bval">{value:.1f}</div></div>')
    out.append("</div>")
    return "".join(out)


def main():
    sonnet = json.loads(pathlib.Path(
        "results/agent-claude-sonnet-5-5-20261114-002532.json").read_text(encoding="utf-8"))
    orders = [w["orders"] for w in sonnet["results"]["agent run 1"]["transcripts"]["3"]]
    agent, ref = trace(3, orders=orders), trace(3, policies.reference)
    curve = json.loads(pathlib.Path("results/training.json").read_text())["curve"]

    score_bars = bars([
        ("Reference", 73.3, REF), ("Small policy, trained", 62.3, SMALL),
        ("Haiku 5.5, run 3", 52.7, AGENT), ("Haiku 5.5, run 2", 48.3, AGENT),
        ("Haiku 5.5, run 1", 42.3, AGENT), ("Sonnet 5.5, run 1", 21.7, AGENT),
        ("Naive pass-through", 0.0, GREY), ("Stop after week 10", 0.0, GREY),
        ("Do nothing", 0.0, GREY)])
    exploit_bars = bars([
        ("Reference", 73.3, REF), ("Do nothing", 0.0, GREY),
        ("Stop after week 1 / 10 / 26 / 40", 0.0, GREY),
        ("Flood every stage with maximum orders", 0.0, GREY),
        ("Hoard stock at the factory", 0.0, GREY), ("Random orders", 0.0, GREY)])
    demand = line_chart([("Demand", INK2, ref["demand"])], height=150, labels=False, width=1120)
    stock = line_chart([("Reference", REF, ref["stock"]), ("Sonnet 5.5", AGENT, agent["stock"])])
    backlog = line_chart([("Reference", REF, ref["backlog"]), ("Sonnet 5.5", AGENT, agent["backlog"])])
    learning = line_chart(
        [("Held-out scenarios", SMALL, [c["held_out"] for c in curve]),
         ("Evaluation scenarios", "#6250d6", [c["evaluation"] for c in curve])],
        width=620, height=300, ymax=100, xlabel="Training episodes (thousands)",
        xs=[f"{c['episodes'] / 1000:g}" for c in curve], label_room=150)

    html = TEMPLATE
    for key, value in {"SCORE_BARS": score_bars, "EXPLOIT_BARS": exploit_bars, "DEMAND": demand,
                       "STOCK": stock, "BACKLOG": backlog, "LEARNING": learning}.items():
        html = html.replace("{{" + key + "}}", value)
    pathlib.Path("report/slides.html").write_text(html, encoding="utf-8")
    print("Wrote report/slides.html")


TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Delayed Supply Chain: demo slides</title>
<style>
@page { size: 13.333in 7.5in; margin: 0; }
* { box-sizing: border-box; }
body { margin: 0; font-family: "Segoe UI", Arial, sans-serif; color: #0b0b0b; background: #fcfcfb; }
.slide { width: 13.333in; height: 7.5in; padding: 0.55in 0.7in 0.45in; page-break-after: always;
         position: relative; overflow: hidden; display: flex; flex-direction: column; }
.slide:last-child { page-break-after: auto; }
h1 { font-size: 30pt; line-height: 1.15; margin: 0 0 6pt; letter-spacing: -0.3pt; max-width: 11.5in; }
.sub { font-size: 15pt; color: #52514e; margin: 0 0 18pt; max-width: 11in; }
.body { flex: 1; display: flex; gap: 0.5in; align-items: flex-start; }
.col { flex: 1; min-width: 0; }
p, li { font-size: 15.5pt; line-height: 1.4; }
ul { margin: 0; padding-left: 20pt; } li { margin-bottom: 8pt; }
.foot { position: absolute; left: 0.7in; right: 0.7in; bottom: 0.25in; font-size: 9.5pt; color: #52514e;
        display: flex; justify-content: space-between; }
.title { justify-content: center; }
.title h1 { font-size: 46pt; } .title .sub { font-size: 20pt; }
.big { font-size: 60pt; font-weight: 700; line-height: 1; font-variant-numeric: tabular-nums; }
.tiles { display: flex; gap: 0.3in; margin-top: 0.3in; }
.tile { flex: 1; border: 1px solid #e4e3df; border-radius: 8px; padding: 14pt 16pt; background: #fff; }
.tile .k { font-size: 12.5pt; color: #52514e; } .tile .s { font-size: 12pt; color: #52514e; margin-top: 4pt; }
.chain { display: flex; align-items: stretch; gap: 0; margin: 6pt 0 16pt; }
.stage { flex: 1; border: 1.5px solid #2a78d6; border-radius: 8px; padding: 10pt 8pt; text-align: center; background: #fff; }
.stage b { font-size: 15pt; display: block; } .stage span { font-size: 11.5pt; color: #52514e; display: block; }
.arrow { width: 0.95in; text-align: center; font-size: 11pt; color: #52514e; align-self: center; line-height: 1.2; }
.arrow i { display: block; font-style: normal; font-size: 20pt; color: #2a78d6; line-height: 0.9; }
.cust { border-style: dashed; border-color: #52514e; }
.bars { width: 100%; } .brow { display: flex; align-items: center; margin-bottom: 9pt; }
.blab { width: 3.3in; font-size: 14pt; } .btrack { flex: 1; background: #f0efec; height: 22pt; border-radius: 0 4px 4px 0; }
.bbar { height: 22pt; border-radius: 0 4px 4px 0; } .bval { width: 0.8in; text-align: right; font-size: 14.5pt; font-variant-numeric: tabular-nums; }
.legend { font-size: 12pt; color: #52514e; margin-top: 6pt; display: flex; gap: 18pt; }
.sw { display: inline-block; width: 11pt; height: 11pt; border-radius: 2px; vertical-align: -1pt; margin-right: 5pt; }
h3 { font-size: 14pt; margin: 0 0 2pt; } .cap { font-size: 11.5pt; color: #52514e; margin: 0 0 4pt; }
svg { width: 100%; height: auto; display: block; }
table { border-collapse: collapse; width: 100%; font-size: 14pt; font-variant-numeric: tabular-nums; }
th, td { text-align: right; padding: 6pt 8pt; border-bottom: 1px solid #e4e3df; }
th:first-child, td:first-child { text-align: left; } th { font-weight: 600; color: #52514e; font-size: 12.5pt; }
code { font-family: Consolas, monospace; font-size: 14pt; background: #f0efec; padding: 2pt 6pt; border-radius: 3px; }
pre { font-family: Consolas, monospace; font-size: 12.5pt; background: #f0efec; padding: 12pt 16pt; border-radius: 6px; margin: 0 0 14pt; line-height: 1.6; }
.quote { border-left: 4px solid #eb6834; padding: 6pt 14pt; font-size: 14pt; color: #0b0b0b; background: #fff; margin: 10pt 0; }
.quote span { display: block; font-size: 11.5pt; color: #52514e; margin-top: 4pt; }
</style></head><body>

<section class="slide title">
  <h1>Delayed Supply Chain</h1>
  <p class="sub">An environment where frontier agents know the rules, see everything,<br>and still run out of stock.</p>
  <p style="font-size:14pt;color:#52514e">AfterQuery "Break the Agent" · Junction 2026 · Team "Rome wasn't built in 48h"</p>
</section>

<section class="slide">
  <h1>The agent runs a four-stage supply chain for 52 weeks</h1>
  <p class="sub">Four order quantities a week. Every decision takes weeks to land.</p>
  <div class="chain">
    <div class="stage"><b>Factory</b><span>makes up to 80 a week</span><span>holding 0.25</span></div>
    <div class="arrow">2 weeks<i>&rarr;</i></div>
    <div class="stage"><b>Distributor</b><span>&nbsp;</span><span>holding 0.50</span></div>
    <div class="arrow">2 weeks<i>&rarr;</i></div>
    <div class="stage"><b>Wholesaler</b><span>&nbsp;</span><span>holding 0.75</span></div>
    <div class="arrow">2 weeks<i>&rarr;</i></div>
    <div class="stage"><b>Retailer</b><span>&nbsp;</span><span>holding 1.00</span></div>
    <div class="arrow">demand<i>&rarr;</i></div>
    <div class="stage cust"><b>Customers</b><span>unfilled demand</span><span>costs 10 a week</span></div>
  </div>
  <div class="body">
    <div class="col"><ul>
      <li><b>Goal:</b> lowest total cost.</li>
      <li><b>Delay:</b> production takes 3 weeks, so a factory decision reaches a customer 9 weeks later.</li>
      <li><b>No undo:</b> orders cannot be cancelled.</li>
    </ul></div>
    <div class="col"><ul>
      <li><b>Demand shifts:</b> one lasting change per episode (step up, step down or ramp), sometimes a season or promotion on top.</li>
      <li><b>The agent sees everything</b> except future demand, and the rules state how demand is generated.</li>
    </ul></div>
  </div>
  <div class="foot"><span>Delayed Supply Chain</span><span>2</span></div>
</section>

<section class="slide">
  <h1>Both Claude models finish well below the reference</h1>
  <p class="sub">Suite score on the five default scenarios. 0 = the naive rule, 100 = knowing the future.</p>
  <div class="body" style="flex-direction:column;gap:0">
    {{SCORE_BARS}}
    <div class="legend"><span><span class="sw" style="background:#2a78d6"></span>Reference run</span>
      <span><span class="sw" style="background:#eb6834"></span>Frontier agent</span>
      <span><span class="sw" style="background:#1baf7a"></span>Policy trained on the environment</span>
      <span><span class="sw" style="background:#9a9993"></span>Baseline</span></div>
    <p style="font-size:13.5pt;color:#52514e;margin-top:10pt">The reference had the lower cost in 19 of 20 agent scenario runs.
      Sonnet was run once; Haiku three times. Both at medium effort, without code execution.</p>
  </div>
  <div class="foot"><span>Delayed Supply Chain</span><span>3</span></div>
</section>

<section class="slide">
  <h1>The agent cuts stock to save money, then cannot serve a demand rise</h1>
  <p class="sub">Sonnet 5.5 and the reference on scenario 3, facing the same demand.</p>
  <div><h3>Customer demand per week</h3>{{DEMAND}}</div>
  <div class="body" style="margin-top:4pt">
    <div class="col"><h3>Stock in the system</h3><p class="cap">Sonnet lets it fall from 311 to 161 while demand is quiet.</p>{{STOCK}}</div>
    <div class="col"><h3>Customer backlog</h3><p class="cap">Empty retailer for 14 weeks, then 115 units short at the end.</p>{{BACKLOG}}</div>
  </div>
  <div class="foot"><span>Cost on this scenario: reference 4,649, Sonnet 10,574</span><span>4</span></div>
</section>

<section class="slide">
  <h1>The mistake is a decision, not missing information</h1>
  <p class="sub">The agents spend less on stock than the reference and far more on unfilled demand.</p>
  <div class="body">
    <div class="col">
      <table>
        <tr><th>Cost per scenario</th><th>Reference</th><th>Haiku 5.5</th><th>Sonnet 5.5</th></tr>
        <tr><td>Holding stock</td><td>3,056–4,569</td><td>2,877–6,020</td><td>2,462–3,964</td></tr>
        <tr><td>Customer backlog</td><td>40–800</td><td>80–4,890</td><td>3,080–6,610</td></tr>
      </table>
      <div class="quote">"There is a large excess, and holding cost is the main concern. […] I'd rather let the retailer run lean."
        <span>Sonnet 5.5, scenario 3, week 15. Ten weeks later the retailer was empty.</span></div>
    </div>
    <div class="col"><ul>
      <li><b>It was told the price.</b> A unit short costs 10; a unit held costs 0.25 to 1.</li>
      <li><b>It was told the risk.</b> The rules say demand may step up by 40–90% between weeks 9 and 35.</li>
      <li><b>It was told the delay.</b> New production needs up to 9 weeks to reach customers, so the stock must already be in the pipeline.</li>
      <li><b>The reference uses the same inputs</b>: the rules text and the same weekly state.</li>
    </ul></div>
  </div>
  <div class="foot"><span>Delayed Supply Chain</span><span>5</span></div>
</section>

<section class="slide">
  <h1>Shortcuts score zero</h1>
  <p class="sub">Suite score for every strategy that avoids the task. 17 automated checks, all passing.</p>
  <div class="body">
    <div class="col" style="flex:1.5">{{EXPLOIT_BARS}}</div>
    <div class="col"><ul>
      <li><b>Stopping early</b> is played out as zero orders, so backlog keeps costing.</li>
      <li><b>Running stock down at the end</b> is closed by the handover rule: finish with nine weeks of demand in the system, or pay 10 per unit short.</li>
      <li><b>Malformed orders</b> count as zero and are capped.</li>
      <li><b>45,000 episodes of direct cost optimisation</b> found no loophole; the result behaves like an ordinary ordering policy and stays below the reference.</li>
    </ul></div>
  </div>
  <div class="foot"><span>Delayed Supply Chain</span><span>6</span></div>
</section>

<section class="slide">
  <h1>A nine-number policy learns the task from reward alone</h1>
  <p class="sub">Stretch goal. Trained on 60 scenarios by cost only; it knows nothing about the demand model.</p>
  <div class="body">
    <div class="col" style="flex:1.25"><h3>Score during training</h3>
      <p class="cap">Scored after every generation on scenarios it never trains on.</p>{{LEARNING}}</div>
    <div class="col">
      <table>
        <tr><th></th><th>Evaluation</th><th>Held out</th></tr>
        <tr><td>Before training</td><td>0.0</td><td>13.5</td></tr>
        <tr><td>After training</td><td>62.3</td><td>71.8</td></tr>
        <tr><td>Reference</td><td>73.3</td><td>76.4</td></tr>
      </table>
      <ul style="margin-top:14pt">
        <li>The reward gives a usable training signal, and what is learned carries over to unseen scenarios.</li>
        <li>It ends above both Claude models, so the task is learnable, not impossible.</li>
        <li>It is a small scripted policy. We did not fine-tune a language model.</li>
      </ul>
    </div>
  </div>
  <div class="foot"><span>Evaluation: seeds 1–5. Held out: seeds 3000–3039. Training: seeds 2000–2059.</span><span>7</span></div>
</section>

<section class="slide">
  <h1>One command reproduces every number</h1>
  <p class="sub">Python standard library only for the simulator, baselines and reference. Deterministic per seed.</p>
  <div class="body">
    <div class="col" style="flex:1.45">
<pre>python evaluate.py                 # reference + baselines
python evaluate.py --agent claude  # three agent runs
python mcp_server.py --seed 1      # MCP server for the harness
python -m unittest -v              # exploit and MCP checks
python train.py                    # retrain the small policy
python replay.py                   # week-by-week replay page</pre>
      <p style="font-size:13.5pt;color:#52514e">Every agent reply is saved with the results, so each failure can be read back.</p>
    </div>
    <div class="col"><h3 style="font-size:16pt;margin-bottom:8pt">What we have not shown</h3><ul>
      <li>Sonnet was run once; five scenarios is a small suite.</li>
      <li>The reference scores only 21 on scenario 1, where demand drifts down and the naive rule is hard to beat.</li>
      <li>Our agent runs used the Anthropic API, not the MCP server.</li>
      <li>The MCP server is tested with the official MCP client, not yet with the AfterQuery harness.</li>
    </ul></div>
  </div>
  <div class="foot"><span>Delayed Supply Chain</span><span>8</span></div>
</section>

</body></html>
"""

if __name__ == "__main__":
    main()
