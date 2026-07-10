# Example: Daily Ops Report

The **100%-roteness archetype**: a scheduled daily report skill in which
*every* step is deterministic — fetch three spreadsheets and one Gmail
label, count rows, apply fixed thresholds, render a report — yet it ran
in production as a full agent loop, re-reasoning the same row-counting
from prose every morning. Graduation eliminates the entire inference
surface: **roteness 1.0, zero `llm_judge` nodes**.

Adapted from a real production skill (companies, people, file IDs, and
internal tool names fictionalized). Its production agent runs averaged
~17 turns and ~0.9M cache-read tokens per run — real money for work
that is 100% arithmetic.

## What this example uniquely demonstrates

- **Graduation can eliminate inference entirely.** BDR (the canonical
  example) shows all five node kinds; this one shows the cleanest win:
  a pipeline whose only non-code element is a human.
- **`hitl_gate` as the sole blocking element** — `manual_data_gate`
  parks the workflow while the duty manager supplies the three numbers
  that live behind interactive dashboards, then the report resumes.
- **The `python` adapter's durable-execution refusal**: emitting this
  pipeline with `--runtime python` fails at emit time (a plain script
  cannot durably park on a human gate) and points at `--runtime dbos`.
- **Mandatory nodes**: 9 of 10 nodes carry `mandatory: true` — the
  dwell/missort thresholds cannot be prompt-drifted away.

## Layout

```
examples/ops-report/
├── README.md              # this file
├── skill/SKILL.md         # input: the source skill
└── expected/pipeline.yaml # the graduated IR (regression baseline)
```

## Try it

```sh
# Report what graduation produces (runs the real graduator agent):
rote analyze examples/ops-report/skill

# Emit runtime code from the committed IR (no agent, instant):
rote emit examples/ops-report/expected/pipeline.yaml --runtime dbos --out /tmp/ops-report

# Watch the python adapter refuse the HITL gate:
rote emit examples/ops-report/expected/pipeline.yaml --runtime python --out /tmp/nope

# Before/after scorecard:
rote eval examples/ops-report/expected/pipeline.yaml
```
