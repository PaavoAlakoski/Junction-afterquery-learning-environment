# Delayed Supply Chain

A reinforcement-learning environment for the AfterQuery "Break the Agent"
challenge. The agent runs a four-stage supply chain for 52 weeks. Goods take
weeks to arrive, demand shifts without warning, and every unit of stock or
unfilled demand costs money.

Requires Python 3.10+. The simulator and baselines use only the standard
library; the Claude agent needs `pip install anthropic`.

## Run it

```
python evaluate.py              # reference run + do-nothing, stop-early and naive baselines
python -m unittest -v           # reward-exploit tests
python evaluate.py --agent claude  # also run Claude three times (see below)
python mcp_server.py --seed 1   # serve one scenario to an agent over MCP
python train.py                 # retrain the small policy (about two minutes)
python replay.py                # build replay.html, a week-by-week comparison
```

Baseline results are saved to `results/baselines.json`; each agent evaluation
gets its own timestamped file in `results/`. Everything is
deterministic for a given seed.

## The task

```
factory (3) -> distributor (2) -> wholesaler (1) -> retailer (0) -> customers
```

Each week the agent chooses four order quantities, one per stage.

- Shipments take 2, 2, 2 and 3 weeks to reach stages 0-3, so a production
  decision reaches a customer nine weeks later.
- A stage that cannot fill an order ships what it has and owes the rest.
  Orders cannot be cancelled, so panic-ordering comes back as a flood.
- Holding stock costs 1.00 / 0.75 / 0.50 / 0.25 per unit per week at stages
  0-3. Unfilled customer demand costs 10 per unit per week.
- Demand is noisy. Every scenario has one lasting change of level (a step up,
  a step down or a ramp) and half also have a temporary disturbance (a
  seasonal wave or a short promotion).
- Handover rule: the chain must end with enough stock in the system to cover
  nine weeks of recent demand; each unit short costs 10. This stops the agent
  from running stock down in the final weeks.

The agent sees everything except future demand: stock at every stage, every
shipment in transit and when it lands, all backorders, full demand and order
history, and all cost parameters. The exact rules text given to the agent is
`supplychain.rules_text()`.

**What an agent can learn here:** acting on the whole pipeline rather than
this week's shortage, telling a lasting demand shift from a blip, and deciding
where in the chain to keep buffer stock.

## Score

For each scenario two anchors are computed on the same demand:

- `naive`: every stage orders what customers bought this week.
- `floor`: a planner that knows future demand exactly.

```
score = 100 * (naive - agent) / (naive - floor), clipped to [0, 100]
```

A suite score pools costs over all scenarios before applying the formula.
Matching the naive rule scores 0; 100 would need knowledge of the future.

## Current results

Default suite, seeds 1-5 (score 0-100, higher is better):

| Policy | Score | Total cost |
|---|---|---|
| Reference | 73.3 | 20,249 |
| Small policy, trained (`train.py`) | 62.3 | 24,586 |
| Claude Haiku 5.5, three runs | 52.7 / 48.3 / 42.3 | 28,390 / 30,114 / 32,473 |
| Claude Sonnet 5.5, one run | 21.7 | 40,622 |
| Naive pass-through | 0.0 | 49,188 |
| Small policy, untrained | 0.0 | 65,260 |
| Stop after week 10 | 0.0 | 834,229 |
| Do nothing | 0.0 | 1,305,699 |

The first, second and last four rows are deterministic and are reproduced by
`python evaluate.py`. The Claude rows are from
`results/agent-claude-haiku-5-5-20261113-234241.json` and
`results/agent-claude-sonnet-5-5-20261114-002532.json`, which hold every
reply. Both models ran at medium effort, received the rules, the full state
and their own notes each week, and had no code execution.

**The gap.** Every agent run finished below the reference on the suite, and
the reference had the lower cost in 19 of 20 agent scenario runs. Haiku's
three runs spread over ten points. Sonnet has been run once, so its exact
score is uncertain.

**Why the agents lose.** They trim stock to save holding cost and are caught
short when demand rises. Their holding cost is about the reference's or
lower; their customer-backlog cost is far higher (reference 40-800 per
scenario, Haiku 80-4,890, Sonnet 3,080-6,610). The rules state that a unit of
backlog costs ten times a unit of retailer stock and that demand may step up
by 40-90%; goods ordered after the rise take up to nine weeks to arrive.
`python replay.py` shows this week by week for any saved run.

Five scenarios is a small sample. On ten further seeds (11-20) the reference
scores 71.4, on 40 other seeds (3000-3039) 76.4, and on 150 more
(1000-1149) 74.2.

## Learning from the environment

`train.py` trains a nine-number ordering policy
(`supplychain/learned.py`) from episode cost alone. The policy knows nothing
about the demand model and does not use the reference's forecast.

| | Evaluation seeds 1-5 | Held-out seeds 3000-3039 |
|---|---|---|
| Before training | 0.0 | 13.5 |
| After 45,000 training episodes | 62.3 | 71.8 |
| Reference | 73.3 | 76.4 |

It trains on seeds 2000-2059 only; the learning curve is in
`results/training.json`. So the reward gives a usable training signal, what
is learned carries over to unseen scenarios, and a policy far smaller than a
language model ends up above both Claude models. It is a small scripted
policy, not a language model: we did not fine-tune one.

## The reference run

`supplychain/policies.py: reference` is an echelon base-stock policy driven by
a Bayesian demand forecast (`supplychain/forecast.py`).

- **Forecast:** the rules tell the agent how demand is generated. The forecast
  draws 4,000 candidate scenarios from that description, weighs each by how
  well it explains the demand seen so far, and reads off the distribution of
  demand over the coming weeks.
- **Ordering:** each stage orders up to a high quantile of forecast demand
  over its cumulative lead time, counting everything already at or on its way
  to that stage and the stages below it.

It uses only the demand history and state the agent also sees. Its service
levels are not tuned on any scenario. They are the newsvendor critical ratios
for a serial supply chain, computed from the costs in the rules:
`(backlog cost + holding cost one stage up) / (backlog cost + retailer holding
cost)`, which gives 0.977, 0.955, 0.932 and 0.909 for stages 0-3.

## Reward integrity

`test_exploits.py` checks that:

- doing nothing, stopping early (after week 1, 10, 26 or 40), flooding the
  chain with maximum orders, and hoarding at the factory all score 0;
- stopping in the last few weeks, or dumping stock and refilling in the final
  week, never beats playing to the end;
- malformed orders (NaN, negatives, strings, wrong length, booleans) count as
  zero and are capped at the limits;
- random orders score 0;
- units are conserved and results are deterministic;
- `mcp_server.py` and `serve.py` give the same cost as playing the
  environment directly; stopping, never acting or sending unreadable input
  is scored as zero orders; the MCP server offers no way to restart or to
  choose the scenario.

The trained small policy is also a search for loopholes: 45,000 episodes of
optimising cost directly ended below the reference, with ordinary ordering
behaviour.

If an agent crashes or stops, the remaining weeks are played with zero orders.

## Running with Harbor

`harbor/tasks/` holds five Harbor tasks, one per scenario (seeds 1-5). In
each, the simulator runs as an MCP server in its own container
(`supply-chain`, streamable HTTP), and the agent's container reaches it only
through the four tools. The simulator's code and the scenario number are not
in the agent's container.

```
harbor run -p harbor/tasks -a oracle                      # reference run
harbor run -p harbor/tasks -a opencode -m <model> -k 3    # three agent runs
harbor run -p harbor/tasks -a nop                         # do-nothing baseline
harbor run -p harbor/tasks -a oracle --ae SOLUTION=stop_early   # stop after week 10
```

Each task's reward is its score divided by 100. Harbor reports rewards per
task, so its average differs from the pooled suite score above:

| | Scenario 1 | 2 | 3 | 4 | 5 | Mean |
|---|---|---|---|---|---|---|
| Reference (oracle) | 0.21 | 0.94 | 0.48 | 0.69 | 0.67 | 0.60 |
| Do nothing, stop early | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

**Status: not yet run under Harbor.** The tasks follow Harbor's published
task format, and every part has been run outside Docker: the MCP server over
HTTP with the official MCP client, and the reference, stop-early and
do-nothing scripts against it. The containers, the `harbor run` commands
above and the OpenCode agent have not been run.

`python harbor/build_tasks.py` regenerates the tasks after a change to the
simulator.

## Connecting an agent

- **MCP (for the judging harness):** `python mcp_server.py --seed 1` serves
  one scenario over stdio; add `--transport http` for
  `http://127.0.0.1:8000/mcp`. It needs only the standard library. Example
  client configuration:

  ```json
  {"mcpServers": {"supply-chain": {
      "command": "python", "args": ["mcp_server.py", "--seed", "1"]}}}
  ```

  The agent gets four tools: `get_rules`, `get_state`,
  `place_orders(retailer, wholesaler, distributor, factory)` and `stop`.
  One server process plays one episode of one scenario; there is no reset
  tool and the agent cannot pick the seed, because replaying a scenario
  would reveal its future demand. Start one server per seed (1-5 for the
  default suite). The server writes the score to
  `results/mcp-seed-<seed>.json` after every week, as the result the episode
  would get if the agent stopped there, so an agent that never acts or stops
  early is scored as placing zero orders from then on.
- **Python:** `SupplyChainEnv(seed)`, then `env.step([q0, q1, q2, q3])`
  until `env.done`. `env.observe()` returns the state as a dict.
- **Any harness:** `python serve.py --seed 1` plays one episode over
  stdin/stdout as JSON lines (protocol in the file's docstring).
- **Claude agent:** set `ANTHROPIC_API_KEY`, then
  `python evaluate.py --agent claude`. It defaults to the cheapest
  model, `claude-haiku-5-5`, at medium effort; set `CLAUDE_MODEL` to
  `claude-sonnet-5-5` or `claude-opus-5-5` for a stronger agent, and
  `CLAUDE_EFFORT` to change effort. A run aborts once estimated spend passes
  `CLAUDE_MAX_USD` (default 2). One
  suite run is 5 x 52 = 260 model calls; the five scenarios run in parallel.
  Try `--seeds 1 --runs 1` first. Full transcripts are saved in the results
  file for failure analysis.
- **Other models:** `python evaluate.py --agent llm` with `LLM_BASE_URL`,
  `LLM_API_KEY` and `LLM_MODEL` set for an OpenAI-compatible endpoint.

## Known limitations

- The reference's score varies a lot by scenario: 21.0, 94.3, 48.3, 69.3 and
  66.6 on seeds 1-5. In scenario 1 demand drifts down, so safety stock held
  against a rise is wasted and the naive rule is close to the best possible
  without foresight; even with service levels chosen in hindsight for that
  scenario, this kind of policy reaches about 22. On the 40 seeds 3000-3039
  it fails to beat the naive rule on 2. One agent scenario run beat it
  (Haiku run 3 on scenario 4).
- The agent results are three Haiku runs and one Sonnet run, at medium effort
  and without code execution. They were made through the Anthropic API with
  `evaluate.py`, where the agent sees the rules, the state and its own notes
  each week. Through the MCP server an agent keeps its whole conversation
  instead; no agent has been run that way yet.
- The reference takes about a second per episode and the tests about half a
  minute, because the forecast is plain Python.
- `floor` is a feasible plan with perfect foresight, not a proven lower bound.
- `mcp_server.py` is tested over stdio with the official MCP Python client
  and over HTTP with the official MCP Python client (full episodes, protocol
  2025-11-25). The Harbor tasks have not been run in Docker; see "Running
  with Harbor".
- In the Harbor tasks the verifier runs in the agent's container after the
  agent has finished, and reads the score from the MCP server. An agent that
  tampered with that container could in principle interfere with the
  reading; Harbor's separate verifier mode would close this.

## Files

```
supplychain/env.py        simulator, demand generator, rules text
supplychain/policies.py   baselines, reference, perfect-foresight anchor
supplychain/forecast.py   Bayesian demand forecast used by the reference
supplychain/scoring.py    cost -> score
supplychain/learned.py    small trainable policy and its trained numbers
supplychain/claude_agent.py  Claude agent (Anthropic SDK)
supplychain/llm_agent.py  OpenAI-compatible LLM agent
evaluate.py               one-command evaluation
train.py                  trains the small policy from episode cost
replay.py                 builds replay.html from saved runs
mcp_server.py             MCP server: one scenario, one episode
harbor/                   Harbor tasks (one per scenario) and their generator
serve.py                  plain JSON-lines adapter (alternative to MCP)
test_exploits.py          reward-integrity and adapter tests
report/                   findings report, slides and demo script
```
