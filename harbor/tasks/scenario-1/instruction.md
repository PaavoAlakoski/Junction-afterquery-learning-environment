# Run the supply chain

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
