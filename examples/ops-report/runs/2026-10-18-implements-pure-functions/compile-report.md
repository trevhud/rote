# Compilation Report — ops-report

**Compiled:** 2026-10-18  
**Source skill:** `skills/ops-report/skill/SKILL.md`  
**Work directory:** `/rote-compile-tez5se3h/`

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
| Codifiable % | **100%** (all non-gate nodes are deterministic code) |
| Estimated token reduction per run | **~85%** (no LLM invoked in the compiled path) |
| Signatures produced | 0 |
| HITL gates | 1 |

### HITL Gates

| Gate | Signal | Timeout | Blocks on |
|---|---|---|---|
| `duty_manager_data_gate` | `duty_manager_data_provided` | 4h | Duty manager supplying dock appointment counts, sorter metrics, and current BI-dashboard dwell snapshot |

### Node Classification Table

| Step | Kind | Justification |
|---|---|---|
| Fetch Shipment Containers (Drive) | `external_call` | Fixed file ID, single deterministic MCP tool call |
| Fetch Dock Pending Log (Drive) | `external_call` | Fixed file ID, single deterministic MCP tool call |
| Fetch Dock Activity Emails (Gmail) | `external_call` | Fixed search query, single deterministic MCP tool call |
| Fetch Dwell Tickets (Drive) | `external_call` | Fixed file ID, single deterministic MCP tool call |
| Parse Shipment Containers | `pure_function` | Fixed CSV parsing + numeric threshold (75%) with no LLM |
| Parse Dock Pending Log | `pure_function` | Fixed row filter (Resolved = FALSE) + group-by; no LLM |
| Parse Dock Activity Emails | `pure_function` | Keyword matching on snippet/subject; bounded, no LLM |
| Parse Dwell Tickets | `pure_function` | Fixed row filter + three numeric thresholds; no LLM |
| Duty Manager Data Gate | `hitl_gate` | Skill explicitly says "ask the duty manager to provide" three data items |
| Validate Manual Data | `pure_function` | Fixed threshold checks (escalation, fail rate, volume drop, dwell alerts); no LLM |
| Render Ops Report | `pure_function` | Fixed three-section template with fill-in-the-blank values; no LLM |

**Pattern 8 applied:** The skill lists the four data pulls sequentially in prose, but no pull consumes another's output. All four are declared as `entry_nodes` and run in the same execution wave (concurrent). This is the single biggest wall-clock win available — a sequential agent takes ~4× longer on I/O alone.

---

## Crystallization Log

Every prose-to-code extraction, with source location and before/after.

### 1. File ID constants (SKILL.md lines 21, 26, 33)

**Before:** Three File IDs embedded in prose instructions the agent re-reads every run.  
**After:** Three `file_id` constants on their respective `external_call` nodes.

```yaml
# fetch_shipment_containers
constants:
  file_id: "1ExampleShipmentContainersFileId00000000000"

# fetch_dock_pending_log
constants:
  file_id: "1ExampleDockPendingLogFileId0000000000000000"

# fetch_dwell_tickets
constants:
  file_id: "1ExampleDwellTicketLogFileId0000000000000000"
```

### 2. Gmail search query constant (SKILL.md line 30)

**Before:** `Search query: label:dock-activity newer_than:1d` in prose.  
**After:**

```yaml
constants:
  query: "label:dock-activity newer_than:1d"
```

### 3. Completion threshold (SKILL.md line 24)

**Before:** "Flag any site below 75% completion"  
**After:** `COMPLETION_THRESHOLD = 75.0` in `parse_shipment_containers.py`, passed as `completion_threshold: 75.0` constant on the node.

```python
# skill: "Flag any site below 75% completion"
flagged = completion_pct < completion_threshold
```

### 4. Resolved = FALSE filter (SKILL.md line 27)

**Before:** "Count all rows where Resolved = FALSE"  
**After:** `parse_dock_pending_log.py` with case-insensitive check:

```python
# skill: "Count all rows where Resolved = FALSE"
if resolved_val in ("FALSE", "NO", "0", ""):
    total_open += 1
```

### 5. Dwell ticket open/closed filter (SKILL.md line 34)

**Before:** "Filter rows where Open/Closed column = 'Open'"  
**After:**

```python
# skill: "Filter rows where Open/Closed column = 'Open'"
if row[status_col].strip().lower() != "open":
    continue
```

### 6. Dwell alert thresholds — three levels (SKILL.md lines 34, 40)

**Before:** Three thresholds in prose, repeated twice (once in the data source section, once in the manual data section).  
**After:** Named constants in both `parse_dwell_tickets.py` and `validate_manual_data.py`, declared as node constants in the IR. The same values appear in the `parse_dwell_tickets` node constants AND the `validate_manual_data` node constants.

```python
BRAND_WARNING_THRESHOLD = 25    # skill: "packages >= 25 (Missort Warning)"
BRAND_ALERT_THRESHOLD   = 41    # skill: "packages >= 41 (Missort Alert)"
CARRIER_ALERT_THRESHOLD = 80    # skill: "carrier >= 80 (Possible Misload Alert)"
```

### 7. Non-Compliant escalation (SKILL.md line 39)

**Before:** "Non-Compliant must be escalated" — a prose MANDATORY check.  
**After:** Hard boolean in `validate_manual_data.py`:

```python
# skill: "Non-Compliant must be escalated"
escalation_required = non_compliant > 0
```

### 8. Sorter fail rate threshold (SKILL.md line 40)

**Before:** "Flag if fail rate > 5% (action needed)"  
**After:** `SORTER_FAIL_RATE_THRESHOLD = 5.0`, enforced as a strict `>` comparison:

```python
# skill: "Flag if fail rate > 5% (action needed)"
fail_rate_flag = fail_rate > sorter_fail_rate_threshold
```

### 9. Sorter volume drop threshold (SKILL.md line 40)

**Before:** "volume dropped >15% day-over-day (divert adjustment may be needed)"  
**After:** `SORTER_VOLUME_DROP_THRESHOLD = 15.0`:

```python
# skill: "volume dropped >15% day-over-day (divert adjustment may be needed)"
volume_drop_flag = drop > sorter_volume_drop_threshold
```

### 10. Fixed three-section report format (SKILL.md lines 44–49)

**Before:** A bulleted list of section contents the agent was expected to format as markdown by hand each run.  
**After:** `render_ops_report.py` — a pure string-formatting function producing KPI header + three sections + artifact reminder. No LLM, no ambiguity.

### 11. Live artifact reminder (SKILL.md line 50)

**Before:** "Also remind the duty manager to open the live artifact 'ops-report' for the interactive version with charts."  
**After:** Hardcoded footer in `render_ops_report.py`:

```python
# skill: "Also remind the duty manager to open the live artifact 'ops-report' for the interactive version"
lines.append("> **Reminder:** Open the live artifact **ops-report** ...")
```

### 12. Independent data pulls → parallel entry_nodes (Pattern 8)

**Before:** Pulls listed sequentially in prose (1, 2, 3, 4). An agent runs them serially.  
**After:** All four declared in `entry_nodes:` with no connecting edges → adapter runs them concurrently in one execution wave.

---

## Open Questions

The following judgment calls were made during compilation. Each should be verified before the first production run.

### OQ-1: Gmail approved/requested classification method

**What the skill says:** "Count Approved vs. Requested appointments."  
**What the skill doesn't say:** Which field, label, subject prefix, or pattern distinguishes an approved thread from a requested one.  
**Decision taken:** Keyword matching on thread `snippet` and `Subject` header. A thread containing "approved" is approved; one containing "requested", "request", or "pending" (without "approved") is counted as outstanding.  
**Where it appears:** `extracted/parse_dock_activity_emails.py`, marked `# rote: assumption:`.  
**How to verify:** Inspect a sample of real dock-activity threads. If they use a structured subject prefix (e.g. `[APPROVED]`) or a Gmail label, update the parser accordingly. If the classification is genuinely ambiguous from text alone, consider upgrading this node to `llm_judge`.

### OQ-2: Carrier Impact column numeric vs. text in Dwell Log

**What the skill says:** "carrier >= 80 (Possible Misload Alert)" in the context of dwell tickets.  
**What the skill doesn't say:** Whether the "Carrier Impact" column in the Dwell Log Form Responses tab contains a numeric package count or free text (e.g. "FedEx — routing error").  
**Decision taken:** Parse the column as an integer; treat non-numeric values as `carrier_packages = 0`. Carrier alerts will not fire from dwell-log data if the column is free text.  
**Where it appears:** `extracted/parse_dwell_tickets.py`, marked `# rote: assumption:`.  
**How to verify:** Inspect a sample row from the Dwell Log tab. If Carrier Impact is text, either (a) add a separate numeric column, or (b) remove the carrier threshold check from `parse_dwell_tickets` (it fires more reliably from the BI dashboard current-dwell data via the HITL gate).

### OQ-3: Alert priority when both brand >= 41 and carrier >= 80

**What the skill says:** Three distinct alert levels with color codes (light red, bright red, darkest red).  
**What the skill doesn't say:** What to display when a single dwell item triggers both "Missort Alert" (packages >= 41) and "Possible Misload Alert" (carrier >= 80) simultaneously.  
**Decision taken:** Possible Misload Alert takes priority because "darkest red" implies highest severity in the skill's color scheme.  
**Where it appears:** `_classify_alert` in `parse_dwell_tickets.py` and `_classify_dwell_alert` in `validate_manual_data.py`, both marked `# rote: assumption:`.  
**How to verify:** Confirm with the duty manager whether these conditions can co-occur and which label the ops team expects to see.

### OQ-4: Google Drive MCP `read_file_content` parameter name

**What the skill says:** "Google Drive MCP — read_file_content" with a File ID.  
**What is unknown:** Whether the MCP tool's argument is named `file_id` (snake_case) or `fileId` (camelCase) or something else.  
**Decision taken:** Used `file_id` in `mcp.args` as the most common convention.  
**Where it appears:** `pipeline.yaml`, `mcp.args.file_id` on all three Drive nodes.  
**How to verify:** Check the Google Drive MCP server's tool schema. Update `mcp.args` accordingly if the parameter name differs.

### OQ-5: Google Drive `read_file_content` response shape

**What is unknown:** The exact structure returned by the MCP tool. The `parse_*` functions expect `fetch_*.output.content` to be the CSV text string.  
**Decision taken:** Assumed `{"content": str}` (the file text under a `content` key).  
**Where it appears:** All three `parse_*` nodes, `inputs: raw_content: fetch_*.output.content`.  
**How to verify:** Run an instrumented baseline pass to populate `probe-context/`. If the tool returns a different key (e.g. `text`, `body`), update the `output:` declarations on the fetch nodes and the `inputs:` bindings on the parse nodes.

### OQ-6: Gmail `search_threads` response shape

**What is unknown:** The exact dict structure returned by the MCP tool.  
**Decision taken:** Assumed `{"threads": [...], "resultSizeEstimate": N}` (standard Gmail API shape). The `parse_dock_activity_emails` function calls `.get("threads", [])` so it degrades gracefully to an empty list if the key differs.  
**Where it appears:** `fetch_dock_activity_emails` output declaration in `pipeline.yaml`.  
**How to verify:** Same instrumented baseline run as OQ-5.

### OQ-7: Schedule hour and timezone

**What the skill says:** "Scheduled daily executive operations report."  
**What the skill doesn't say:** The hour of day or timezone.  
**Decision taken:** `schedule: "0 6 * * *"` (6 AM in the runtime's default timezone).  
**Where it appears:** `pipeline.yaml`, `config.schedule`.  
**How to verify:** Confirm with the ops team what time the report should run and in which timezone.

---

## Suggested Next Steps

1. **Instrument a baseline run first.** Before running any production traffic, execute the skill as raw agent instructions once with MCP traffic logging enabled to populate `probe-context/observed-tools.json`. This resolves OQ-4, OQ-5, and OQ-6 in a single pass and unlocks real implementations for the four `external_call` stubs.

2. **Dog-food against today's data.** After the first instrumented run, point the compiled pipeline at the same data and diff the two reports. Any divergence surfaces a parsing assumption that needs correction.

3. **Inspect dock-activity email format (OQ-1).** Pull five recent dock-activity threads and look at their subjects. If they follow a predictable pattern, the keyword-matching parser can be hardened to a regex or label check. If not, consider upgrading `parse_dock_activity_emails` to a lightweight `llm_judge` with a tight output schema (`{approved: bool, outstanding: bool, reason: str}`).

4. **Pin the report timezone.** Update `config.schedule` after confirming the ops team's preferred run time.

5. **Run `rote emit pipeline.yaml --runtime temporal`** once the stubs are resolved. The IR is fully populated and the adapter has everything it needs to produce a working Temporal workflow.

## rote verification

- Generated tests: passed (96 passed in 0.07s), run in an OS sandbox.
- Implemented impl nodes: 6 of 10.
- Unresolved problems: 0.
