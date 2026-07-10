# Example: Deal Monitor

The **data-heavy archetype**: a scheduled morning dashboard that pulls
an entire Slack intake channel plus 60 days of Gmail quoting threads
into context, classifies every thread, scores every new deal, and
generates a full HTML dashboard — as an agent, the most expensive shape
there is (large tool results re-read as cache tokens on every turn;
tens of thousands of output tokens re-deriving the same HTML each run).

Adapted from a real production skill (companies, people, channel IDs,
and paths fictionalized). Its production agent runs averaged ~22 turns
and ~1.6M cache-read tokens per run; the graduated pipeline replaces
that with two parallel fetches, three schema-constrained judges, and a
template render. **Roteness 0.75** — the remaining 25% is genuinely
fuzzy work (freeform Slack prose → structured fields; reading email
threads; rubric scoring) and correctly *stays* LLM.

This is also the calibration fixture for the estimator's payload-aware
"before" cost model (`tokens_per_external_call_result`): its measured
per-turn transcript growth (~6k tokens) anchored the default.

## What this example uniquely demonstrates

- **Parallel entry waves** — the fixed Gmail searches fire concurrently
  with the Slack pull + parse (the agent ran them sequentially).
- **Fan-out judges** — `classify_thread_step` runs per matched thread
  and `score_new_opportunities` per unscored deal (`fan_out: true`).
- **Deterministic gating of inference** — `filter_opportunities` is a
  `mandatory: true` pure function (region rules + SKU threshold) that
  bounds what ever reaches the judges.
- **External-dependency handling** — the source skill reads a separate
  deal-scoring skill at runtime; the graduator turned that dangling
  reference into an optional `scoring_rubric` pipeline input instead of
  silently inlining a guess.
- **Template render replacing LLM generation** — the HTML dashboard is
  a pure function, deleting the largest output-token cost in the skill.

## Layout

```
examples/deal-monitor/
├── README.md              # this file
├── skill/SKILL.md         # input: the source skill
└── expected/pipeline.yaml # the graduated IR (regression baseline)
```

## Try it

```sh
# Report what graduation produces (runs the real graduator agent):
rote analyze examples/deal-monitor/skill

# Emit runtime code from the committed IR (no agent, instant):
rote emit examples/deal-monitor/expected/pipeline.yaml --runtime dbos --out /tmp/deal-monitor

# Before/after scorecard (the payload-aware estimator at work):
rote eval examples/deal-monitor/expected/pipeline.yaml
```
