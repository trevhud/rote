# Eval scorecard — expense-audit

Generated: 2026-10-18T20:09:37Z · static estimate (nothing was executed)

## Before → after

| Metric | As agent instructions | As compiled pipeline | Change |
|---|---|---|---|
| Wall clock (compute) | 3.2 min – 6.5 min | 2 s | −99% (189.9× less) |
| LLM tokens touched | 460k–1.1M | 0 | eliminated |
| Sampled output tokens | 3.8k–7.5k | 0 | eliminated |
| Steps decided by a sampled LLM | 9 of 9 | 0 of 8 (0 schema-constrained) | — |
| Roteness (deterministic steps) | 0% | 100% | — |
| Human approval gates | implicit (chat back-and-forth) | 1 durable (`controller_review`) | — |

## Cost per run (live official prices)

| Model | Tier | Before | After | Change |
|---|---|---|---|---|
| claude-fable-5-1 | flagship | $0.75 – $1.27 | $0.000 | eliminated |
| claude-sonnet-5-5 | mid | $0.17 – $0.31 | $0.000 | eliminated |
| claude-haiku-5-5 | small | $0.011 – $0.021 | $0.000 | eliminated |

Prices fetched 2026-10-18T18:15:37Z from: https://models.dev/api.json (cross-checked against https://openrouter.ai/api/v1/models). Never hardcoded.

## Pipeline node detail

| Node | Kind | Calls | LLM in/call | LLM out/call | Wall s/call |
|---|---|---|---|---|---|
| `convert_to_usd` | pure_function | 1 | — | — | 0.01 |
| `find_duplicates` | pure_function | 1 | — | — | 0.01 |
| `find_late_submissions` | pure_function | 1 | — | — | 0.01 |
| `group_by_employee` | pure_function | 1 | — | — | 0.01 |
| `audit_report` | pure_function | 1 | — | — | 0.01 |
| `summarize_audit` | pure_function | 1 | — | — | 0.01 |
| `controller_review` | hitl_gate | 1 | — | — | 0 |
| `file_report` | external_call | 1 | — | — | 1.5 |

## Assumptions (audit me)

- Turn estimate: compiler sidecar (eval.yaml).
- Token counting: chars/3.8 approximation.
- Agent cost model: cache-aware transcript growth — turn *i* re-reads min(C₀ + (i−1)·Δ, cap) context, where the cap models harness compaction near the context window; fresh tokens bill at cache-write price, re-reads at cache-read price.
- HITL gate waits are human time and excluded from wall-clock on both sides.
- Priors (calibrated on the BDR runs in `examples/bdr-outreach/runs/`; every value overridable):
  - `seconds_per_turn` = 13.0
  - `output_tokens_per_turn` = 250.0
  - `transcript_growth_per_turn` = 900.0
  - `transcript_cap_tokens` = 165000.0
  - `system_overhead_tokens` = 16000
  - `tokens_per_external_call_result` = 6000.0
  - `payload_tokens_per_tool` = {}
  - `turns_per_step_low` = 1.0
  - `turns_per_step_high` = 2.5
  - `llm_ttft_seconds` = 1.2
  - `llm_output_tokens_per_second` = 60.0
  - `tokens_per_input_field` = 60.0
  - `tokens_per_output_field` = 40.0
  - `external_call_seconds` = 1.5
  - `pure_function_seconds` = 0.01
  - `agent_loop_turns_per_iteration` = 3.0
  - `agent_loop_default_max_iterations` = 10
  - `chars_per_token` = 3.8

## Measured (2 trials per side)

| Metric | As agent instructions | As compiled pipeline |
|---|---|---|
| Successful runs | 2/2 | 2/2 |
| Wall clock (s) | 42.9 (min 40.1, max 45.6) | 9.8 (min 5.6, max 14.0) |
| Cost per run (USD) | 0.218 (list-price notional) | 0.0000 (measured judge usage, live prices) |
| Agent turns | 6.0 (min 6.0, max 6.0) | — |
| Identical outputs | 1/2 | 2/2 |

Agent-side fields that diverged across runs:
- `report_path`: 50% of runs agree

Suggested prior re-fits from these runs (not auto-applied):
- `seconds_per_turn` ≈ 7.14
- `output_tokens_per_turn` ≈ 607.1
- `transcript_growth_per_turn` ≈ 3405.4

