# Demo script (10 minutes)

Slides: `report/slides.pdf`. Live page: `replay.html` in the project root
(open it in a browser beforehand; it needs no network).

## Before you go on

1. In a terminal in the project folder, run `python evaluate.py` once so the
   output is on screen, and leave the terminal open.
2. Open `replay.html`. Choose **Scenario 3** and the **Sonnet** run.
3. Open `slides.pdf` full screen.
4. Ask AfterQuery the two open questions at the bottom of this page, if you
   have not already.

## Running order

| Time | Slide | Who shows what | Say |
|---|---|---|---|
| 0:00 | 1 | Title | "Our agent knows the rules, sees the whole system, and still runs out of stock. We'll show you why." |
| 0:30 | 2 | The task | Four stages, four orders a week. A factory decision reaches a customer nine weeks later. Stock costs up to 1 a week; an unfilled order costs 10. Demand shifts once per episode, and the rules say so. |
| 2:00 | 3 | The gap | Reference 67. Haiku 42 to 53 over three runs. Sonnet 22 in one run. Doing nothing and stopping early score 0. Say plainly that Sonnet was run once. |
| 3:00 | 4, then `replay.html` | Why | Switch to the live page. Point at weeks 13 to 26: demand is quiet, Sonnet lets stock fall from 311 to 161. Then week 25 onwards: the retailer is empty for 14 weeks. Hover to show the numbers. Switch scenario or run if a judge asks. |
| 5:00 | 5 | It is a fair failure | Read the quote. The agent was told the price of being short, the size of the possible rise, and the delay. The reference uses the same rules text and the same weekly state. |
| 6:00 | 6 | Reward integrity | Every shortcut scores 0. The handover rule closes the "run stock down at the end" trick. Invite the judges to try to break it. |
| 7:00 | 7 | Learning | A nine-number policy, trained only on cost, goes from 0 to 62 and holds up on scenarios it never saw. Say plainly that it is not a language model. |
| 8:00 | terminal | Reproduce | Show the `python evaluate.py` output. Mention `python -m unittest -v` (17 checks, under a minute) and `python mcp_server.py --seed 1` for the harness. |
| 8:45 | 8 | Limits | Name the four limits yourself before a judge does. |
| 9:30 | | Close | "The skill being tested is planning under delay: paying a small, certain cost now to avoid a large, uncertain one later. Both models know the numbers and still make the wrong trade." |

## Numbers to have ready

| | Score | Total cost |
|---|---|---|
| Reference | 67.0 | 22,754 |
| Small policy, trained | 62.3 | 24,586 |
| Haiku 5.5 (three runs) | 52.7 / 48.3 / 42.3 | 28,390 / 30,114 / 32,473 |
| Sonnet 5.5 (one run) | 21.7 | 40,622 |
| Naive pass-through | 0.0 | 49,188 |
| Stop after week 10 | 0.0 | 832,895 |
| Do nothing | 0.0 | 1,305,699 |

- The reference had the lower cost in 18 of 20 agent scenario runs.
- Reference on 40 unseen scenarios (seeds 3000-3039): 78.1.
- Customer-backlog cost per scenario: reference 0-320, Haiku 80-4,890,
  Sonnet 3,080-6,610.

## Likely questions

**Why did the stronger model do worse?**
Sonnet trimmed stock harder than Haiku. That lowered its holding cost and
raised its backlog cost by more. It is one run, so we do not claim Sonnet is
reliably worse than Haiku, only that both are well below the reference.

**Is the reference using anything the agent does not have?**
No. It reads the demand model from the same rules text and uses the same
weekly state. Its four service levels were tuned on seeds 1000-1149 and
checked on 1150-1249; the evaluation uses seeds 1-5.

**The reference scores 0 on scenario 1. Why?**
Demand drifts down in that scenario, so the stock the reference keeps against
a rise is never needed, and the naive rule is hard to beat. This happens on
about one scenario in seven. The suite score pools all scenarios, and on 40
unseen scenarios the reference scores 78.1.

**How does the harness connect?**
`python mcp_server.py --seed N` is an MCP server (stdio, or HTTP with
`--transport http`) with four tools: `get_rules`, `get_state`,
`place_orders` and `stop`. One process plays one episode of one scenario.
The score is written to `results/mcp-seed-N.json` after every week.

**Why is there no reset tool?**
Demand is fixed per scenario, so an agent that could restart would play the
second time knowing the future. One process, one episode closes that.

**Were your agent runs made through the MCP server?**
No. They used the Anthropic API directly: each week the agent saw the rules,
the state and its own notes. Through MCP it keeps its whole conversation. We
expect the same failure, but we have not measured it.

**Can the score be gamed?**
Doing nothing, stopping at any point, flooding, hoarding and random orders
all score 0. Stopping late or dumping stock at the end never beats playing
on, because of the handover rule. Malformed orders count as zero. We also ran
45,000 episodes of direct cost optimisation and it found ordinary ordering
behaviour, not a loophole.

**Why is 100 not reachable?**
The top anchor is a planner that knows future demand. No agent can know
that, so 100 marks a ceiling, not a target.

**Is five scenarios enough?**
It is a small suite, chosen to keep one agent evaluation cheap (260 model
calls). `python evaluate.py --seeds 11 12 13 ...` runs any others.

**How realistic is the simulation?**
It is the standard serial supply chain from the inventory literature: one
product, fixed delivery times, full visibility. Real chains are messier. We
kept it clean on purpose so that the difficulty comes from the decision and
nothing is hidden.

**Did you do the stretch goal?**
In part. We trained a small policy on experience from the environment and
showed that it improves and carries over to unseen scenarios. We did not
fine-tune a language model.

## Settled with AfterQuery

- They want an MCP server: `mcp_server.py`.
- The agent does not need code execution.

## Still to check before the demo

- **Run the Harbor tasks once on a machine with Docker.** `harbor run -p
  harbor/tasks -a oracle` should give rewards 0.00, 0.90, 0.33, 0.65, 0.65.
  They have never been run in containers.
- **Compare `harbor/tasks/` with AfterQuery's scoring skeleton and starter
  kit.** We built from Harbor's public docs, not from their skeleton.
- **Run OpenCode on at least one task** with the team's model credits, so
  the gap is measured under the judging harness.
- **Ask a mentor how they run the stop-early baseline**, and whether a
  reference that scores 0 on one of five tasks is a concern for them.
