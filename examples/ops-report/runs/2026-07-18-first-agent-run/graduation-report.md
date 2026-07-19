# Graduation Report — ops-report

**Source skill:** `examples/ops-report/skill/SKILL.md`
**Graduated:** ops-report v0.1.0
**Graduator:** rote-graduate (manual / Claude Sonnet 4.6)

---

## Summary Metrics

| Metric | Value |
|--------|-------|
| Total nodes | 13 |
| `external_call` | 4 |
| `pure_function` | 8 |
| `hitl_gate` | 1 |
| `llm_judge` | 0 |
| `agent_loop` | 0 |
| Codifiable (non-HITL) | **100%** (12 / 12) |
| Estimated token reduction per run | ~95% |

### HITL Gates

| Gate | Signal | Timeout | Blocks On |
|------|--------|---------|-----------|
| `duty_manager_data_gate` | `duty_manager_data_provided` | 4h | Three manual data items from the duty manager: dock appointment counts, sorter metrics per site, and current dwell from the BI dashboard |

### MCP Requirements

| Server | Nodes | Tool |
|--------|-------|------|
| `google_drive` | `fetch_shipment_containers`, `fetch_dock_pending_log`, `fetch_dwell_tickets` | `read_file_content` |
| `gmail` | `fetch_dock_emails` | `search_threads` |

---

## Phase 2 — Node Classification Table

| Step | Node ID | Kind | Justification |
|------|---------|------|---------------|
| Fetch shipment containers spreadsheet | `fetch_shipment_containers` | `external_call` | Fixed GDrive MCP call with hardcoded file ID; no runtime variation |
| Fetch dock pending log | `fetch_dock_pending_log` | `external_call` | Fixed GDrive MCP call with hardcoded file ID |
| Search Gmail dock-activity threads | `fetch_dock_emails` | `external_call` | Fixed Gmail MCP call with hardcoded query string |
| Fetch open dwell ticket log | `fetch_dwell_tickets` | `external_call` | Fixed GDrive MCP call with hardcoded file ID and tab name |
| Parse facility summary + flag < 75% | `parse_shipment_containers` | `pure_function` | Deterministic tabular parse + numeric threshold check; no LLM |
| Count Resolved=FALSE, group by site/type | `parse_dock_pending_log` | `pure_function` | Deterministic filter + group-by; no LLM |
| Count Approved vs Requested emails | `parse_dock_emails` | `pure_function` | Keyword matching on subject/snippet; deterministic |
| Filter Open tickets + apply 25/41/80 | `parse_dwell_tickets` | `pure_function` | Deterministic filter + numeric threshold classification |
| Request manual data from duty manager | `duty_manager_data_gate` | `hitl_gate` | Skill explicitly says 'ask the duty manager to provide' these three items |
| Apply Non-Compliant escalation rule | `apply_dock_appointment_rules` | `pure_function` | Fixed rule: Non-Compliant → escalation_required; fully deterministic; MANDATORY |
| Apply sorter metric thresholds | `apply_sorter_rules` | `pure_function` | Numeric threshold checks (5%, 15%); deterministic |
| Apply BI dwell color-coded thresholds | `apply_dwell_thresholds` | `pure_function` | Numeric thresholds (25, 41, 80); deterministic |
| Assemble executive report | `assemble_report` | `pure_function` | Fixed template with prescribed sections; no LLM generation needed |

**Why zero `llm_judge` nodes:** every classification in this skill is numeric threshold-based (25/41/80 packages, 75% completion, 5% fail rate, 15% volume drop) rather than rubric-based. No free-text interpretation is required. This is unusual for a real-world skill and reflects a highly structured operations domain.

**Why zero `agent_loop` nodes:** all data sources are identified by hardcoded file IDs and search queries. There is no exploratory tool use, no backfilling, and no variable-path iteration.

---

## Phase 3 — Crystallization Log

All crystallizations come from `examples/ops-report/skill/SKILL.md`.

| # | Source (file:prose) | Extracted To | Constants / Logic |
|---|---------------------|-------------|-------------------|
| 1 | SKILL.md line ~21: "Flag any site below 75% completion" | `shipment_parser.py:COMPLETION_THRESHOLD_PCT` | `COMPLETION_THRESHOLD_PCT = 75.0` |
| 2 | SKILL.md line ~25: "Count all rows where Resolved = FALSE" | `dock_pending_parser.py:parse_dock_pending_log` | filter: `_is_unresolved()` |
| 3 | SKILL.md line ~25: "Break down by site...and issue type" | `dock_pending_parser.py:parse_dock_pending_log` | group-by: `site_map[site][issue_type]` |
| 4 | SKILL.md line ~28: "Search query: label:dock-activity newer_than:1d" | `gmail_ops.py:GMAIL_DOCK_ACTIVITY_QUERY` | `GMAIL_DOCK_ACTIVITY_QUERY = "label:dock-activity newer_than:1d"` |
| 5 | SKILL.md line ~31: "Tab: Dwell Log - Form Responses" | `gdrive.py:DWELL_TICKET_TAB` | `DWELL_TICKET_TAB = "Dwell Log - Form Responses"` |
| 6 | SKILL.md line ~32: "Filter rows where Open/Closed column = 'Open'" | `dwell_ticket_parser.py:parse_dwell_tickets` | filter: `status == "Open"` |
| 7 | SKILL.md line ~33: "Flag packages >= 25 (Missort Warning)" | `dwell_ticket_parser.py:PKG_WARNING_THRESHOLD` | `PKG_WARNING_THRESHOLD = 25` |
| 8 | SKILL.md line ~33: ">= 41 (Missort Alert)" | `dwell_ticket_parser.py:PKG_ALERT_THRESHOLD` | `PKG_ALERT_THRESHOLD = 41` |
| 9 | SKILL.md line ~33: "carrier >= 80 (Possible Misload Alert)" | `dwell_ticket_parser.py:CARRIER_ALERT_THRESHOLD` | `CARRIER_ALERT_THRESHOLD = 80` |
| 10 | SKILL.md line ~38: **"Non-Compliant must be escalated"** | `manual_data_rules.py:apply_dock_appointment_rules` | `mandatory: true` + `escalation_required = non_compliant > 0` |
| 11 | SKILL.md line ~39: "Flag if fail rate > 5% (action needed)" | `manual_data_rules.py:SORTER_FAIL_RATE_THRESHOLD_PCT` | `SORTER_FAIL_RATE_THRESHOLD_PCT = 5.0` |
| 12 | SKILL.md line ~39: "volume dropped >15% day-over-day" | `manual_data_rules.py:SORTER_VOLUME_DROP_THRESHOLD_PCT` | `SORTER_VOLUME_DROP_THRESHOLD_PCT = 15.0` |
| 13 | SKILL.md line ~40: "Brand 25-40 = Missort Warning (light red), 41+ = Missort Alert (bright red)" | `manual_data_rules.py:apply_dwell_thresholds` | same 25/41 constants, brand-severity classification |
| 14 | SKILL.md line ~40: "Carrier 80+ missing = Possible Misload Alert (darkest red)" | `manual_data_rules.py:apply_dwell_thresholds` | same 80 constant, carrier-severity classification |
| 15 | SKILL.md line ~21, ~25, ~31: Three hardcoded Google Drive file IDs | `gdrive.py:SHIPMENT_CONTAINERS_FILE_ID` etc. | Three file-ID string constants |
| 16 | SKILL.md lines ~44-48: Fixed three-section report format with KPI header | `report_assembler.py:assemble_report` | Markdown template function replacing all free-form LLM report generation |

**Highest-value extraction:** #10 (Non-Compliant escalation). In the original skill, this MANDATORY rule is enforced only by the phrase "Non-Compliant must be escalated" in the prose. In the graduated pipeline, `apply_dock_appointment_rules` is `mandatory: true` and `escalation_required = True` whenever `non_compliant > 0` — the rule cannot be forgotten or skipped regardless of model drift, prompt edits, or long trajectories.

**Second-highest value:** #16 (report assembler). The original skill's agent generates the entire three-section markdown from scratch on every run (~2-4 agent turns of token-expensive generation). The `assemble_report` pure function replaces all of that with a template call.

### Before / After Snippet — Non-Compliant Escalation

**Before (prose in SKILL.md):**
```
Dock Appointment Completion (from the dock scheduler's appointment view for
the prior day): count of Green (Complete), Red (No Call/No Show), Grey
(Canceled), and any other color (Non-Compliant). Non-Compliant must be
escalated.
```

**After (extracted code):**
```python
MANDATORY: true (IR flag — cannot be made conditional)

def apply_dock_appointment_rules(dock_appointments: DockAppointmentCounts):
    escalation_required = dock_appointments.non_compliant > 0
    ...
```

---

## Open Questions

1. **Day-over-day sorter volume comparison.**
   The skill says "volume dropped >15% day-over-day" but the duty manager is only asked to provide today's volume. Where does yesterday's volume come from?
   - **Decision made:** `SorterSiteData.prior_volume` is an optional field in the HITL gate output. The duty manager can provide prior-day volume explicitly, or the pipeline can persist yesterday's value in a store.
   - **Reviewer action:** Decide whether to add a storage layer (`rote.state` or a simple database write) to carry yesterday's sorter metrics forward, or ask the duty manager to provide it manually each day. If the latter, add a note to the HITL gate's notification template.

2. **GDrive MCP return format.**
   The skill names `read_file_content` as the MCP tool but does not specify what format it returns for Sheets files. The parser functions assume a `list[dict]` with string keys matching column headers. If the actual MCP tool returns a different shape (e.g., `{rows: [...], headers: [...]}` or raw CSV), the parsers need adjustment.
   - **Reviewer action:** Run `fetch_shipment_containers()` against a test Sheet and inspect the response shape before deploying parsers.

3. **Gmail classification heuristic.**
   `parse_dock_emails` classifies threads using keyword matching on `subject` / `snippet`. The actual subject-line conventions used by your dock scheduling system may differ from the keywords `approved`, `requested`, `pending` used in the stub.
   - **Reviewer action:** Sample 10-20 real dock-activity emails and verify the keywords match, or replace the keyword classifier with a regex pattern that matches your actual subject templates.

4. **HITL signal payload schema.**
   The `duty_manager_data_gate` output schema defines `DockAppointmentCounts`, `SorterSiteData`, and `DwellInputRecord`. The signal delivery mechanism (Slack bot command, web form, direct API call) must produce a JSON payload matching these types. This is adapter-specific; the IR defines the contract.
   - **Reviewer action:** Choose a HITL delivery mechanism (e.g., a Slack slash command or a web form) and verify the payload schema matches what the adapter emits for this gate.

5. **Carrier vs. brand package count.**
   The skill says "Carrier 80+ missing = Possible Misload Alert." In `apply_dwell_thresholds`, the carrier threshold is applied to the same `packages` field as the brand threshold because the BI dashboard record has a single `packages` field. If the BI dashboard reports brand packages and carrier packages separately, the schema needs a `carrier_packages` field.
   - **Reviewer action:** Confirm whether the BI dashboard dwell export has a single package count or separate brand/carrier package counts, and adjust `DwellInputRecord` accordingly.

---

## Suggested Next Steps

### Dogfood first
Run the graduated pipeline against yesterday's real data by providing the three Google Drive file IDs and confirming the GDrive MCP tool is authenticated. This validates all four `external_call` nodes and the parsing logic before connecting the HITL gate.

### Evals to run
- Unit-test all eight `pure_function` nodes in isolation with fixture data. The threshold logic in `parse_dwell_tickets` and `apply_dock_appointment_rules` is the most safety-critical — add edge cases for the exact threshold boundaries (24 packages = OK, 25 = warning, 40 = warning, 41 = alert).
- Integration-test the `assemble_report` function with a full fixture payload to verify markdown formatting and that the artifact reminder appears.

### Nodes to revisit after first production run
- `parse_dock_emails` — the keyword classifier is the most likely node to need adjustment. Monitor the `outstanding_requests` list after the first week to see if false positives or missed approvals occur.
- `apply_dwell_thresholds` — if the BI dashboard export includes separate brand and carrier package counts, split the `DwellInputRecord` schema and adjust `apply_dwell_thresholds` accordingly (open question #5 above).

### Adapters
This pipeline requires durable execution (`requires_durable_execution = True`) because of the `duty_manager_data_gate`. The `python` adapter will refuse it at emit time. Recommended runtimes in order of setup complexity:
- **DBOS (default):** simplest; state in SQLite/Postgres; `rote emit pipeline.yaml --runtime dbos`
- **Temporal:** good for teams already running Temporal; `rote emit pipeline.yaml --runtime temporal`
- **Inngest:** good for teams with an existing Next.js/Node service; `rote emit pipeline.yaml --runtime inngest`

### Schedule
The pipeline is configured for `"0 6 * * 1-5"` (06:00 weekdays). Adjust the cron expression in `pipeline.yaml` for your timezone and whether weekend ops reports are needed.
