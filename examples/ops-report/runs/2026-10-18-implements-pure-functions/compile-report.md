# ops-report Compilation Report

## Summary Metrics

| Metric | Value |
|--------|-------|
| Total nodes | 12 |
| `pure_function` | 6 |
| `external_call` | 4 |
| `llm_judge` | 1 |
| `hitl_gate` | 1 |
| `agent_loop` | 0 |
| Codifiable % | 91% (10 of 11 non-gate nodes) |
| Estimated token reduction per run | ~75–85% |

**Why 91% is justified:** ops-report is almost entirely a structured data transformation. All four data sources are fixed-schema spreadsheets or a deterministic email search. Every threshold is numeric. The output format is fully specified in the skill. The single `llm_judge` (email thread classification) is the only genuinely fuzzy step. No research, no exploratory tool use, no variable procedure.

### HITL Gates

| Gate | Signal | Timeout | What it blocks |
|------|--------|---------|----------------|
| `duty_manager_data_gate` | `duty_manager_data_provided` | 4h | Dock appointment completion counts, sorter metrics, current dwell — three datasets that live behind interactive dashboards and cannot be auto-pulled |

---

## Node Classification Table

| Node | Kind | Justification |
|------|------|---------------|
| `fetch_shipment_containers` | `external_call` | Fixed MCP call with constant file ID; no LLM reasoning on call shape |
| `parse_shipment_containers` | `pure_function` | Deterministic CSV parse + numeric threshold flag (< 75%); all logic fully stateable |
| `fetch_dock_pending_log` | `external_call` | Fixed MCP call with constant file ID |
| `parse_dock_pending_log` | `pure_function` | Row count where Resolved=FALSE + groupby; pure aggregation |
| `fetch_dock_emails` | `external_call` | Fixed Gmail MCP search with constant query string |
| `classify_dock_email` | `llm_judge` | Email thread prose is unbounded; classification (approved/requested/other) requires reading content — pre-filter on subject lines handles clean cases, LLM handles ambiguous replies |
| `aggregate_dock_emails` | `pure_function` | Counting and set-matching on typed classification results; deterministic |
| `fetch_dwell_tickets` | `external_call` | Fixed MCP call with constant file ID + tab name |
| `parse_dwell_tickets` | `pure_function` | Filter Open rows + three numeric thresholds (25/41/80); fully deterministic |
| `duty_manager_data_gate` | `hitl_gate` | Skill explicitly says "ask the duty manager to provide" — data lives behind interactive dashboards with no API |
| `apply_manual_thresholds` | `pure_function` | Five numeric threshold rules applied to typed dict inputs; no LLM |
| `render_ops_report` | `pure_function` | Skill specifies exact report structure (KPI header + 3 sections); pure string formatting |

---

## Crystallization Log

### Pattern 2 — Fixed constants extracted from prose

| Source | Prose | Extracted constant | Module |
|--------|-------|--------------------|--------|
| SKILL.md line 23 | "Flag any site below 75% completion" | `COMPLETION_THRESHOLD_PCT = 75.0` | `extracted/parsers.py` |
| SKILL.md line 33 | "Flag packages >= 25 (Missort Warning)" | `MISSORT_WARNING_PACKAGES = 25` | `extracted/parsers.py` |
| SKILL.md line 33 | "Flag packages >= 41 (Missort Alert)" | `MISSORT_ALERT_PACKAGES = 41` | `extracted/parsers.py` |
| SKILL.md line 33 | "carrier >= 80 (Possible Misload Alert)" | `MISLOAD_ALERT_CARRIER = 80` | `extracted/parsers.py` |
| SKILL.md line 41 | "Flag if fail rate > 5% (action needed)" | `SORTER_FAIL_RATE_THRESHOLD_PCT = 5.0` | `extracted/thresholds.py` |
| SKILL.md line 41 | "volume dropped >15% day-over-day" | `SORTER_VOLUME_DROP_THRESHOLD_PCT = 15.0` | `extracted/thresholds.py` |
| SKILL.md lines 40-42 | "Brand 25-40 = Missort Warning, 41+ = Missort Alert, Carrier 80+ = Possible Misload Alert" | `DWELL_WARNING_PACKAGES=25`, `DWELL_ALERT_PACKAGES=41`, `DWELL_MISLOAD_CARRIER=80` | `extracted/thresholds.py` |

All seven numeric thresholds now live as module-level constants. A silent prose edit that changes "75%" to "70%" no longer goes unnoticed — the constant requires a code commit that shows up in review.

### Pattern 3 — MANDATORY checks moved to code

The skill has five implicit-mandatory checks enforced only by prose; all are now `mandatory: true` nodes or enforced inside mandatory `pure_function` nodes:

| Prose rule | Where codified | mandatory: true? |
|------------|---------------|-----------------|
| "Flag any site below 75% completion" | `parse_shipment_containers` → `flagged_sites` field | yes |
| "Non-Compliant must be escalated" | `apply_manual_thresholds` → `non_compliant_escalated` flag | yes |
| "Flag if fail rate > 5% (action needed)" | `apply_manual_thresholds` → `fail_rate_flagged` per site | yes |
| "volume dropped >15% day-over-day (divert adjustment may be needed)" | `apply_manual_thresholds` → `volume_drop_flagged` per site | yes |
| All dwell alert thresholds (25/41/80) | `parse_dwell_tickets` + `apply_manual_thresholds` → `alert_level` fields | yes (both nodes) |

Before compilation: every one of these rules was prose a future model could silently skip. After compilation: the flagging logic is in code, tested, and the `mandatory: true` flag prevents the adapter from making any of these nodes conditional.

### Pattern 4 — Fixed report template extracted

The skill specifies an exact output structure:
```
Before (SKILL.md lines 44-50):
"Produce an executive-level summary with:
 - KPI header: Active sites, Dock Pending count, Avg Completion %, Open Dwell Tickets
 - Section 1 — Dock Compliance: Appointment tiles + Gmail summary + Pending Log table
 - Section 2 — Missort & Loss: Sorter metrics + container flow table by site with completion %
 - Section 3 — Dwell: BI-dashboard dwell alerts (color-coded) + Open Dwell Impact Log tickets"

After: extracted/report.py:render_ops_report — a pure_function that takes typed inputs
from all upstream nodes and produces the markdown string deterministically.
No LLM token is spent on formatting; the LLM is freed for genuinely fuzzy work.
```

### Pattern 8 — Independent data pulls parallelized

The skill lists the four data sources sequentially, but none depends on another:

```
Before (prose ordering implies sequential):
  1. Fetch Shipment Containers
  2. Fetch Dock Pending Log
  3. Fetch Dock Activity Emails
  4. Fetch Dwell Tickets

After (pipeline.yaml):
  entry_nodes: [fetch_shipment_containers, fetch_dock_pending_log,
                fetch_dock_emails, fetch_dwell_tickets]
  # → all four run in the same wave, in parallel
```

Wall-clock time for the data-pull phase drops from ~4× one call to ~1× one call.

### Email aggregation split into two nodes

The skill says "Count Approved vs. Requested appointments. Flag any outstanding requests not yet approved." In an agent, this is one prompt pass over all threads. Compiled:

- `classify_dock_email` (llm_judge, fan_out) — one bounded LLM call per thread; output schema is tight (enum + ref + thread_id)
- `aggregate_dock_emails` (pure_function) — deterministic counting and set-matching on the typed outputs

The split means the LLM only decides per-thread classification (bounded); the counting and outstanding-request detection are code.

---

## Phase 6 Validation Checklist

- [x] Every `pure_function` / `external_call` has an `impl:` pointing at a real file
- [x] All four MCP tool calls carry `mcp:` bindings (`google_drive/read_file_content` × 3, `gmail/search_threads` × 1)
- [x] All `pure_function` nodes are implemented (no stubs); all `external_call` nodes are Regime-1 stubs with descriptive `NotImplementedError` messages
- [x] Every `pure_function` is called from `tests/test_*.py`
- [x] `classify_dock_email` has both `signature:` (legacy Temporal path) and `signature_spec:` (cross-runtime)
- [x] `classify_dock_email` `inputs:` key (`thread`) matches `signature_spec.input_schema.required` (`[thread]`)
- [x] `duty_manager_data_gate` has `signal: duty_manager_data_provided`
- [x] All nodes in `entry_nodes`, `exit_nodes`, `edges.from`, `edges.to` exist in `nodes:`
- [x] No YAML `on:` keys (using `on_signal:` and `retry_on:` as specified)
- [x] No `temperature:` in `signature_spec` (not set — correct per llm-judge-extraction.md)

---

## Open Questions

### 1. Email structure unknown — classify_dock_email may over-use LLM

**Judgment call:** classified as `llm_judge`. If dock-activity emails have consistent subject line structure (e.g., always "Dock Appointment Approved: ..." vs. "Dock Appointment Request: ..."), the pre-filter in `ClassifyDockEmail.forward()` handles them without the LLM. Run the eval set against real Gmail threads before deploying. If the pre-filter hit rate is >90%, consider promoting to `pure_function` (pattern match on subject + body keywords).

### 2. File IDs are example placeholders

The skill specifies:
- `1ExampleShipmentContainersFileId00000000000`
- `1ExampleDockPendingLogFileId0000000000000000`
- `1ExampleDwellTicketLogFileId0000000000000000`

These are clearly placeholder values. Constants are in `extracted/google_drive.py` at the module level. **Replace before first production run.**

### 3. Spreadsheet column names are assumed

`parse_shipment_containers`, `parse_dock_pending_log`, and `parse_dwell_tickets` find columns case-insensitively by trying several candidate names (e.g., `"completion %"`, `"completion%"`, `"complete %"`). The skill doesn't state the exact column names — the parsers will raise `ValueError` naming the absent column on first run if the headers don't match any candidate. Add the observed header name to the candidate list in that case.

### 4. Week-over-week trend column name unknown

`parse_shipment_containers` attempts several candidate column names (`"prior week completion %"`, `"wow %"`, etc.) and silently omits WoW data if none match. The skill says "show week-over-week trend if available" — the exact column name is not stated. First production run will reveal it; add it to the candidates list in `extracted/parsers.py`.

### 5. duty_manager_data_gate output format

The HITL gate waits for `duty_manager_data_provided` signal. The output schema declares three typed fields (`dock_appointment_completion`, `sorter_metrics`, `current_dwell`), but the skill doesn't specify whether the duty manager will provide these as structured JSON (via a form) or as natural-language text that needs parsing. **The adapter team should define a structured input form** matching the gate's output schema before wiring the signal handler. If natural language is inevitable, add an `llm_judge` node between the gate and `apply_manual_thresholds` to parse it.

### 6. Sorter metrics: prev_day_volume source

The volume drop threshold (> 15% day-over-day) requires `prev_day_volume` in each sorter site entry. The skill says the duty manager provides "volume output and fail rate % per site" but does not explicitly mention prior-day volume. The gate output schema includes `prev_day_volume: int | None`. If the duty manager can't provide it, `volume_drop_flagged` will always be `False`. Verify with the duty manager team what data they have access to.

---

## Suggested Next Steps

1. **Replace placeholder file IDs** in `extracted/google_drive.py` with real Google Drive file IDs before any test run.

2. **Run `classify_dock_email` eval set** (`evals/classify_dock_email.jsonl`) against real Gmail `dock-activity` threads. If the subject-line pre-filter handles >90% of real cases, promote the node to `pure_function` and remove the LLM dependency entirely.

3. **Wire the HITL gate signal** by defining a structured input form for `duty_manager_data_provided`. The gate output schema in `pipeline.yaml` is the form spec; the adapter generates the signal handler from it.

4. **First production run smoke test**: after replacing file IDs, run the pipeline with real data. Expect `ValueError` messages from parsers on the first run if any column names don't match candidates — add the observed names to the candidate lists in `extracted/parsers.py`.

5. **Add sorter metrics golden fixtures** once real data is available. The `apply_manual_thresholds` tests currently use hand-crafted data; golden fixtures from a real run will catch edge cases (fractional fail rates, missing sites, etc.).

6. **Point rote-compile at itself** once the first production run validates the pipeline — the dogfood run will surface any remaining crystallization opportunities in the compilation infrastructure itself.
