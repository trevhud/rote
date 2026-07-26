# Expected Compilation Output for BDR

This directory holds the **golden output** for compiling the BDR outreach
skill — the hand-written ground truth that the rote compiler should
eventually produce automatically. It exists for two reasons:

1. **It drives the IR schema design.** Writing the `pipeline.yaml` by hand
   forces every IR concept (node kinds, edges, HITL gates, retry policies,
   loop bodies) to be tested against a real complex skill *before* the
   schema is formalized in Python.
2. **It is the regression suite for the compiler.** Once `rote compile`
   exists, its output will be diffed against this directory. Substantive
   differences either reveal a bug in the compiler or update the golden
   output (with a commit explaining why).

## Contents

| File / dir | Purpose |
|---|---|
| `pipeline.yaml` | The IR — runtime-agnostic DAG of nodes, edges, gates |
| `extracted/` | Pure-function Python modules for `pure_function` and `external_call` nodes |
| `signatures/` | Typed DSPy/BAML signatures for `llm_judge` nodes |
| `evals/` | Seed eval sets for each `llm_judge` node, harvested from the skill's rubric |
| `runtimes/temporal/` | Temporal-specific emitted code (Workflow + Activities) |
| `compile-report.md` | Human-readable summary (created later) |

## Status

- [x] `pipeline.yaml` — hand-drafted from the BDR skill
- [ ] `extracted/` modules — stubs only
- [ ] `signatures/` modules — stubs only
- [ ] `evals/` — not started
- [ ] `runtimes/temporal/` — not started

The hand-drafted `pipeline.yaml` is the v0 source of truth. Everything else
in this directory will be filled in as the corresponding rote subsystems
land.
