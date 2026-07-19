# Graduation Report — ops-report

**Graduated:** ops-report v0.1.0
**Source skill:** `examples/ops-report/skill/SKILL.md`

---

## Summary Metrics

| Metric | Value |
|---|---|
| Total nodes | 11 |
| `external_call` | 4 |
| `pure_function` | 6 |
| `hitl_gate` | 1 |
| `llm_judge` | 0 |
| `agent_loop` | 0 |
| Mandatory nodes | 10 of 11 (hitl_gate is implicit mandatory) |
| Codifiable nodes (non-HITL) | **10 / 10 = 100%** |
| Estimated token reduction | **~95–100%** (no LLM inference in graduated pipeline) |
| Parallel speedup | Wave 1 runs 4 fetches concurrently vs. sequentially in agent |

### HITL Gates

| Gate | Signal | Timeout | Blocks on |
|---|---|---|---|
| `duty_manager_data_gate` | `duty_manager_data_submitted` | 4h | Duty manager providing dock appointment completion, sorter metrics, and BI dwell data before the report can be assembled |

---

## Crystallization Log

Every prose-to-code extraction from Phase 3, with before/after.

### 1. Fixed Google Drive file IDs → constants

**Before (SKILL.md prose):**
```
File ID: 1ExampleShipmentContainersFileId00000000000
File ID: 1ExampleDockPendingLogFileId0000000000000000
File ID: 1ExampleDwellTicketLogFileId0000000000000000
Tab: Dwell Log - Form Responses
```

**After (`extracted/gdrive.py`):**
```python
SHIPMENT_CONTAINERS_FILE_ID = "1ExampleShipmentContainersFileId00000000000"
DOCK_PENDING_LOG_FILE_ID = "1ExampleDockPendingLogFileId0000000000000000"
DWELL_TICKET_LOG_FILE_ID = "1ExampleDwellTicketLogFileId0000000000000000"
DWELL_TICKET_TAB = "Dwell Log - Form Responses"
```

These IDs cannot drift through prompt edits. A change requires a code commit.

---

### 2. Fixed Gmail query → constant

**Before (SKILL.md prose):**
```
Search query: label:dock-activity newer_than:1d
```

**After (`extracted/gmail.py`):**
```python
DOCK_ACTIVITY_QUERY = "label:dock-activity newer_than:1d"
```

---

### 3. Completion threshold → constant + mandatory pre-filter

**Before (SKILL.md prose):**
> Flag any site below 75% completion.

**After (`extracted/parsers.py`):**
```python
COMPLETION_WARNING_THRESHOLD = 0.75  # flag sites below 75% completion
```
Applied in `parse_shipment_containers()` — the flag cannot be skipped or reinterpreted.

---

### 4. Dwell alert thresholds → constants + mandatory checks

**Before (SKILL.md prose):**
> Flag packages >= 25 (Missort Warning), >= 41 (Missort Alert), carrier >= 80 (Possible Misload Alert).
> Brand 25-40 = Missort Warning (light red), 41+ = Missort Alert (bright red).
> Carrier 80+ missing = Possible Misload Alert (darkest red).

**After (`extracted/parsers.py` and `extracted/thresholds.py`):**
```python
DWELL_MISSORT_WARNING_PACKAGES = 25
DWELL_MISSORT_ALERT_PACKAGES = 41
DWELL_MISLOAD_ALERT_CARRIER_PACKAGES = 80
```
The spreadsheet thresholds (25, 41) are applied in `parse_dwell_tickets()`.
The BI dashboard threshold (80, carrier) is applied in `validate_manager_data()` since
it applies to a different data source that arrives via the HITL gate.

---

### 5. Sorter alert thresholds → constants + mandatory checks

**Before (SKILL.md prose):**
> Flag if fail rate > 5% (action needed) or if volume dropped >15% day-over-day (divert adjustment may be needed).

**After (`extracted/thresholds.py`):**
```python
SORTER_FAIL_RATE_THRESHOLD = 0.05     # > 5%  → action needed
SORTER_VOLUME_DROP_THRESHOLD = 0.15   # > 15% DoD → divert adjustment
```
Both flags are computed deterministically in `validate_manager_data()`.

---

### 6. Non-Compliant escalation → MANDATORY node

**Before (SKILL.md prose):**
> Non-Compliant must be escalated.

**After (`extracted/thresholds.py`, node `validate_manager_data`, `mandatory: true`):**
```python
# validate_manager_data() sets:
non_compliant_escalation_required = data['appointment_completion']['non_compliant_count'] > 0
```
This node is `mandatory: true` in the IR — the adapter emits it as an unconditional
activity. The escalation check can never be silently skipped by an agent distracted by
other content in a long context window.

---

### 7. Report format → fixed template function

**Before (SKILL.md prose):**
```
Produce an executive-level summary with:
- KPI header: Active sites, Dock Pending count, Avg Completion %, Open Dwell Tickets
- Section 1 — Dock Compliance: ...
- Section 2 — Missort & Loss: ...
- Section 3 — Dwell: ...
Also remind the duty manager to open the live artifact "ops-report"...
```

**After (`extracted/report.py`):**
```python
def assemble_report(*, run_date, shipment_data, dock_pending_data,
                    dock_activity, dwell_ticket_data, validated_manager_data) -> str:
    # Pure template rendering — no LLM. The output format is fully specified.
```
The LLM was doing string formatting. The graduated pipeline uses zero tokens for this.

---

### 8. Independent data pulls → concurrent entry nodes (Pattern 8)

**Before (SKILL.md prose):**
> "1. Shipment Containers … 2. Dock Pending Log … 3. Dock Activity Emails … 4. Open Dwell Tickets"
> (listed sequentially — an agent would run them one at a time)

**After (pipeline.yaml):**
```yaml
entry_nodes:
  - fetch_shipment_containers    # \
  - fetch_dock_pending_log       #  | all run concurrently in wave 1
  - fetch_dock_activity_emails   #  |
  - fetch_dwell_tickets          # /
```
None of the four fetches produces output needed by another fetch. Sequential ordering
was an artifact of agent execution. The graduated pipeline runs all four in parallel,
cutting wall-clock latency to the slowest single fetch (~30s) vs. ~2 minutes sequential.

---

## Open Questions

### OQ-1 — Gmail thread classification method

The skill says "Count Approved vs. Requested appointments" from
`label:dock-activity` threads. The classification method is not specified: the
threads might have structured subjects like `[APPROVED] Dock appt 14:00 EAST1` or
they might require reading the email body.

**Decision taken:** classified as `pure_function` (deterministic parsing).

**Reviewer should verify:** what format do dock-activity emails actually use? If the
approval/request status is in a structured subject-line prefix or a thread label, the
pure_function classification holds. If the body is free text requiring judgment, this
node needs to become `llm_judge` with a typed signature.

---

### OQ-2 — Google Drive MCP return format

`read_file_content` on a Google Sheet can return different formats depending on the
MCP implementation (plain CSV, TSV, HTML export, or plain text). The extracted stubs
assume the content is parseable as tabular data with named columns.

**Reviewer should verify:** what format does your Google Drive MCP actually return?
If it returns HTML, the stubs need an HTML table parser. If CSV/TSV, a csv.reader
works. Document the format in the stub's docstring once confirmed.

---

### OQ-3 — HITL gate signal payload structure

The `duty_manager_data_gate` expects the duty manager to submit a structured payload
with `appointment_completion`, `sorter_metrics`, and `current_dwell` fields. In the
current SKILL.md there is no specification for how the manager delivers this data
(form, Slack message, email reply).

**Decision taken:** modeled as a standard HITL gate with a named signal. The runtime
adapter's `notify` message template tells the manager what to provide, but the
delivery mechanism is left to the operator to configure.

**Reviewer should verify:** what UX is the duty manager actually using? If it's a Slack
slash command or a structured form, the signal payload schema should match that form's
output exactly.

---

### OQ-4 — Week-over-week trend availability

The skill says "Show week-over-week trend if available." This implies the spreadsheet
sometimes lacks the prior-week column. The `parse_shipment_containers` stub returns
`wow_trend: str | None` with None when unavailable.

**Decision taken:** `pure_function` — the parser handles the optional column gracefully.
No LLM needed to decide "is the data available?" — it's a column existence check.

---

### OQ-5 — Carrier vs. brand dwell threshold split

The 80-package threshold applies to "carrier" dwell from the BI dashboard (duty manager
data), while the 25/41 thresholds apply to "brand" packages in the Dwell Ticket Log
spreadsheet. These come from different data sources and are therefore handled in two
different nodes (`parse_dwell_tickets` and `validate_manager_data`).

**Reviewer should verify:** confirm this split is correct. If the carrier threshold also
applies to the spreadsheet's dwell tickets, move `DWELL_MISLOAD_ALERT_CARRIER_PACKAGES`
into `parsers.py` and apply it there as well.

---

## Suggested Next Steps

### 1. Fill in the stubs (highest priority)

The four extracted parse functions are `NotImplementedError` stubs. Run a sample Google
Drive export of each spreadsheet and confirm the column names and format, then implement:
- `parse_shipment_containers()` — facility table parser
- `parse_dock_pending_log()` — filter + group parser
- `parse_dock_activity()` — Gmail thread classifier (see OQ-1)
- `parse_dwell_tickets()` — open-ticket filter with thresholds

### 2. Implement `validate_manager_data()`

The threshold logic is fully specified. Implement it and write pytest cases for each
threshold boundary (24 pkgs → normal, 25 → warning, 40 → warning, 41 → alert, 80 carrier → misload).

### 3. Implement `assemble_report()`

The format is fully specified in SKILL.md. This is string template work — write it once
and add a snapshot test against a known-good fixture.

### 4. Run `rote emit` to get deployable code

```sh
rote emit /tmp/rote-graduate-unhgh7uk/pipeline.yaml --runtime dbos --out /tmp/ops-report-app
```

The HITL gate makes this skill require a durable runtime — the `python` adapter will
refuse it with a clear error pointing at `--runtime dbos`.

### 5. Wire the Slack notify channel

Set `notify_channel` in the pipeline input (or hard-code it in the extracted config)
to the channel where the duty manager expects the daily report request.

### 6. First production run: dogfood OQ-1 and OQ-2

Run a real graduation against the actual Drive spreadsheets and Gmail threads. The
stub implementations will surface the real data format (OQ-2) and the Approved vs.
Requested classification method (OQ-1) immediately on the first real run.

### 7. Node to regraduate first

`parse_dock_activity` — if OQ-1 reveals that email subject lines don't carry
structured status info, this node upgrades from `pure_function` to `llm_judge`
with a small seed eval set. All other nodes are pure computation and unlikely to need
revisiting.
