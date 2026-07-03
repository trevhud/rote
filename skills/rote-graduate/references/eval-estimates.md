# Eval estimates — the `eval.yaml` sidecar

You have just read the source skill more carefully than anyone ever
will. Capture one extra artifact while that understanding is fresh: an
estimate of what the skill costs to run **as raw agent instructions**,
step by step. This powers `rote eval`'s before/after scorecard — the
number that tells the user what graduation bought them.

Write it to `eval.yaml`, next to `pipeline.yaml`. It is a sidecar, not
part of the IR: it describes how the *source skill* behaves when an
agent executes it, not how the graduated pipeline behaves.

## What you are estimating

For each step you identified in Phase 2, estimate how many **agent
turns** that step consumes when the skill runs as instructions in an
agent loop. One turn = one assistant message, usually containing one
tool call. Ranges, not points — a step that "usually takes 2 calls but
sometimes 5" is `{low: 2, high: 5}`.

Ground rules:

- **Count from the skill's own text.** If the prose says "make four
  parallel lookup calls", that's 4 tool calls but likely 1–2 turns
  (agents batch parallel calls). If it says "repeat until you have 25
  qualified contacts", estimate the realistic iteration count from any
  hints (quota sizes, batch sizes, typical yields), not the worst case.
- **Include the overhead the prose hides.** Reading reference files,
  re-checking rubric sections, correcting malformed tool calls — a
  realistic agent spends turns on these. A step whose happy path is 2
  calls is rarely under `{low: 2, high: 3}` in practice.
- **Exploration is where the range widens.** A deterministic step
  (fixed API call, fixed template) is a tight range. A research or
  search step is a wide one — say `{low: 3, high: 10}` — and that
  honesty is the point.
- **`totals` may be less than the sum.** If steps interleave or share
  context (the agent researches while it vets), give an explicit
  `totals` block with your whole-run judgment. Omit `totals` when the
  steps are genuinely sequential and the sum is right.

## Schema

```yaml
version: 1
source_skill: examples/bdr-outreach/skill
steps:
  - description: "Research target company and drug across data sources"
    node_id: target_research        # the IR node this step became, if 1:1
    phase: "1"
    estimated_turns: {low: 4, high: 10}
    estimated_tool_calls: {low: 5, high: 14}
  - description: "Resolve ZoomInfo taxonomy IDs (4 lookup calls)"
    node_id: taxonomy_lookup
    phase: "1"
    estimated_turns: {low: 1, high: 2}
    estimated_tool_calls: {low: 4, high: 4}
totals:                             # optional — omit if the sum is right
  low: 28
  high: 45
notes: >-
  Lead generation dominates: the loop re-queries ZoomInfo with adjusted
  filters until the quota is met; typical campaigns converge in 3-5
  iterations of 2-3 turns each.
```

Field reference:

| Field | Required | Meaning |
|---|---|---|
| `version` | yes | Always `1` |
| `source_skill` | no | Path to the skill you graduated |
| `steps[].description` | yes | The step, in the skill's own vocabulary |
| `steps[].node_id` | no | IR node id, only when the mapping is clean 1:1 |
| `steps[].phase` | no | Source skill phase, same convention as the IR |
| `steps[].estimated_turns` | yes | `{low, high}` agent turns |
| `steps[].estimated_tool_calls` | no | `{low, high}` tool calls, when distinct from turns |
| `totals` | no | Whole-run `{low, high}`; omit to mean "sum the steps" |
| `notes` | no | The one paragraph a reader needs to trust the numbers |

Validated by `rote.eval.sidecar.EvalEstimates` — if you can run Python,
`load_eval_estimates("eval.yaml")` is the dry check, same pattern as
`load_pipeline`.

## Calibration anchors (BDR reference points)

From real runs of BDR-scale skills (7 phases, ~20 distinct steps,
heavy tool use):

- Whole-run turn counts landed in the **30–57 turn** range.
- Wall clock was **~13 seconds per turn** including tool execution.
- A single fixed API call step: 1–2 turns. A batch-and-review step:
  2–4. A bounded search loop: 6–15. A genuinely open research step:
  4–10.

If your step estimates sum to something wildly outside the anchor range
for a comparably sized skill, re-examine before writing — either the
skill really is bigger/smaller (fine, say so in `notes`) or a step
estimate is off.

## What NOT to do

- **Do not estimate the graduated pipeline's cost.** `rote eval`
  computes that from the IR directly; your sidecar covers only the
  before side.
- **Do not put estimates in `pipeline.yaml`.** The IR stays
  runtime-agnostic and behavior-only; estimates live in the sidecar.
- **Do not give false precision.** `{low: 3, high: 3}` claims
  certainty. Reserve it for steps the skill fully pins down (fixed call
  counts, fixed batches).
- **Do not skip steps.** Every Phase 2 step appears exactly once, even
  the trivial ones — a reader reconciles your sidecar against the
  classification table.
