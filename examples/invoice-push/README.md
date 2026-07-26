# invoice-push — the agent-loop archetype

Adapted from a real production skill (all names, URLs, and identifiers
fictionalized): a browser-automation job that bulk-pushes imported
carrier invoices from a fulfillment platform's imports screen into a
procurement portal, then writes a three-tab report to Google Drive.

## Why this example exists

This is the **`agent_loop` archetype** — the one node kind the other
examples don't exercise. The per-row cycle (read the row, check
eligibility, click ⋮ → Push, interpret the toast, refresh and
re-locate by Batch ID) can't be a pure function or a single external
call: it's a bounded loop of browser actions with data-dependent
branching, including two toast results (`ERR-AUTH`, `ERR-CONN`) that
abort the entire run. The compiled pipeline keeps exactly that one
step as a bounded `agent_loop` (with a five-node `loop_body`, a
termination condition, and a 200-iteration cap) while everything
around it — date-window math, filter setup, toast interpretation,
report assembly, Drive I/O — compiles to deterministic code.
Roteness: 12 of 13 steps.

## Why the eval sidecar matters here

The production original of this skill is the fixture that forced the
loop-aware cost model. Its two measured runs took **184 and 730 agent
turns** (the per-row loop dominates everything), while a sidecar
without iteration awareness estimated 40–110. The committed
[`expected/eval.yaml`](expected/eval.yaml) shows the corrected form:
per-row steps declare `iterations: {low, high}` and their whole-run
turn estimate multiplies out to 85–639 — bracketing the first measured
run and landing within ~12% of the second (vs. 5–18× under without
iterations). Cost regime: *turn-dominated* (hundreds of cheap turns), the
opposite of deal-monitor's *payload-dominated* profile (few turns,
heavy data).

## Layout

- [`skill/SKILL.md`](skill/SKILL.md) — the source skill, as an agent
  would run it
- [`expected/pipeline.yaml`](expected/pipeline.yaml) — the compiled
  IR (13 nodes: 1 agent_loop + 5 loop-body sub-nodes, 4 more external
  calls, 3 pure functions)
- [`expected/eval.yaml`](expected/eval.yaml) — the eval sidecar with
  per-row `iterations`, the calibration fixture for loop-dominated
  skills
