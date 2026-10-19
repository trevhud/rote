# ops-report — Compilation Report

**Compiled:** 2026-10-18  
**Source skill:** `skills/ops-report/skill/SKILL.md`  
**Work directory:** `rote-compile-homj7048/`

---

## Summary Metrics

| Metric | Value |
|---|---|
| Total nodes | 13 |
| `pure_function` | 8 |
| `external_call` | 4 |
| `llm_judge` | 0 |
| `agent_loop` | 0 |
| `hitl_gate` | 1 |
| Codifiable nodes (non-HITL) | 12 / 12 = **100%** |
| HITL gates | 1 (`request_manual_data_gate`) |
| LLM calls in compiled pipeline | **0** |

**Estimated token cost reduction per run:** 12–22 agent turns eliminated (≈ 24,000–44,000 tokens/run). At daily cadence this is roughly 8–16 M tokens/year. The LLM is not in the hot path at all — the entire report is produced by deterministic code after one HITL pause.

**HITL gates and what they block:**
- `request_manual_data_gate` — blocks report render until the duty manager supplies three manual data sets: dock appointment completion color counts, sorter metrics per site, and current BI dashboard dwell data. Timeout: 4 h.

---

## Phase 2: Node Classification

| Step | Kind | Justification |
|---|---|---|
| Fetch shipment containers (Drive MCP) | `external_call` | Fixed file ID, fixed tool, no reasoning needed; retry/timeout semantics required |
| Fetch dock pending log (Drive MCP) | `external_call` | Same as above, different file ID |
| Fetch dock activity emails (Gmail MCP) | `external_call` | Fixed search query `label:dock-activity newer_than:1d`; deterministic call |
| Fetch dwell ticket log (Drive MCP) | `external_call` | Fixed file ID and tab name |
| Parse shipment containers | `pure_function` | CSV parse + threshold comparison (75%); fully deterministic on typed inputs |
| Parse dock pending log | `pure_function` | Row filter (Resolved=FALSE) + group-by; purely deterministic |
| Parse dock activity emails | `pure_function` | Keyword count over thread subjects; deterministic given the input |
| Parse dwell tickets | `pure_function` | Row filter + three numeric threshold comparisons |
| Request manual data gate | `hitl_gate` | Skill explicitly says "ask the duty manager to provide" — real human input required |
| Apply appointment flags | `pure_function` | Pure boolean (other > 0 → escalate); fully deterministic |
| Apply sorter flags | `pure_function` | Two numeric comparisons (> 5%, > 15%); no judgment needed |
| Apply dwell thresholds | `pure_function` | Tiered numeric threshold on package count; no judgment needed |
| Render report | `pure_function` | Fixed three-section template with KPI header; the skill shows the exact structure |

**Parallelism win (Pattern 8):** All four data-pull steps share no outputs — the skill lists them sequentially only because an agent executes one tool call at a time. In the compiled pipeline they all fire in wave 1 simultaneously (`entry_nodes` lists all four). Typical wall-clock speedup over a raw agent: 3–4× on the data-pull phase.

---

## Phase 3: Crystallization Log

Every extraction below moves a MANDATORY data rule from prose into code.

### Extraction 1 — Completion threshold (Pattern 2 + Pattern 3)
**Source:** `SKILL.md:21` — "Flag any site below 75% completion"  
**Before:** The LLM reads this rule and applies it to each row in its head.  
**After:**
```python
# extracted/parsers.py
COMPLETION_THRESHOLD: float = 75.0
# ...
flagged = completion_pct < completion_threshold  # skill: "Flag any site below 75% completion"
```
**Constant lifted to IR:** `parse_shipment_containers.constants.completion_threshold: 75.0`

---

### Extraction 2 — Dwell package thresholds (Pattern 2 + Pattern 3)
**Source:** `SKILL.md:33` — "Flag packages >= 25 (Missort Warning), >= 41 (Missort Alert), carrier >= 80 (Possible Misload Alert)"  
**Before:** LLM applies three nested rules to every open ticket row.  
**After:**
```python
# extracted/parsers.py
DWELL_PACKAGES_WARNING: int = 25
DWELL_PACKAGES_ALERT: int = 41
DWELL_CARRIER_ALERT: int = 80

def _classify_dwell_packages(packages, *, packages_warning_threshold,
                              packages_alert_threshold, carrier_alert_threshold) -> str:
    if packages >= carrier_alert_threshold:   # skill: "carrier >= 80 (Possible Misload Alert)"
        return "Possible Misload Alert"
    if packages >= packages_alert_threshold:  # skill: ">= 41 (Missort Alert)"
        return "Missort Alert"
    if packages >= packages_warning_threshold: # skill: ">= 25 (Missort Warning)"
        return "Missort Warning"
    return ""
```
**Constants lifted to IR:** `parse_dwell_tickets.constants.{packages_warning_threshold: 25, packages_alert_threshold: 41, carrier_alert_threshold: 80}`  
Same three constants also lifted to `apply_dwell_thresholds.constants.{brand_warning_threshold: 25, brand_alert_threshold: 41, carrier_alert_threshold: 80}`.

---

### Extraction 3 — Resolved = FALSE filter (Pattern 3)
**Source:** `SKILL.md:26` — "Count all rows where Resolved = FALSE"  
**Before:** LLM counts rows by reading prose instruction.  
**After:**
```python
resolved_val = row[resolved_col].strip().upper()
if resolved_val in ("FALSE", "0", "NO", "N"):  # skill: "Count all rows where Resolved = FALSE"
    total_open += 1
```

---

### Extraction 4 — Non-Compliant escalation (Pattern 3 — MANDATORY)
**Source:** `SKILL.md:38` — "Non-Compliant must be escalated"  
**Before:** Prose rule; an agent could miss it under prompt drift.  
**After:**
```python
# extracted/thresholds.py
other = int(appointment_completion.get("other", 0))
# skill: "Non-Compliant must be escalated"
"escalation_required": non_compliant_count > 0
```
**IR flag:** `apply_appointment_flags.mandatory: true` — this node cannot be made conditional.

---

### Extraction 5 — Sorter fail-rate flag (Pattern 2 + Pattern 3)
**Source:** `SKILL.md:39` — "Flag if fail rate > 5% (action needed)"  
**Before:** LLM applies this per-site rule.  
**After:**
```python
SORTER_FAIL_RATE_THRESHOLD: float = 5.0
if fail_rate_pct > fail_rate_threshold:   # skill: "Flag if fail rate > 5% (action needed)"
    flags.append("ACTION NEEDED")
```
**Constant lifted to IR:** `apply_sorter_flags.constants.fail_rate_threshold: 5.0`

---

### Extraction 6 — Sorter volume-drop flag (Pattern 2 + Pattern 3)
**Source:** `SKILL.md:39` — "volume dropped >15% day-over-day (divert adjustment may be needed)"  
**Before:** LLM computes day-over-day % and compares.  
**After:**
```python
SORTER_VOLUME_DROP_THRESHOLD: float = 15.0
drop_pct = (prev_volume - volume) / prev_volume * 100.0
if drop_pct > volume_drop_threshold:  # skill: "volume dropped >15% day-over-day..."
    flags.append("DIVERT ADJUSTMENT MAY BE NEEDED")
```
**Constant lifted to IR:** `apply_sorter_flags.constants.volume_drop_threshold: 15.0`

---

### Extraction 7 — BI dashboard dwell thresholds (Pattern 2 + Pattern 7)
**Source:** `SKILL.md:40` — "Brand 25-40 = Missort Warning (light red), 41+ = Missort Alert (bright red). Carrier 80+ missing = Possible Misload Alert (darkest red)"  
**Before:** LLM reads rubric and classifies each row.  
**After:** Same `_classify_dwell_row()` helper as Extraction 2, lifted to `extracted/thresholds.py`.

---

### Extraction 8 — Fixed report template (Pattern 4)
**Source:** `SKILL.md:44–48` — exact report structure with section names, KPI header columns, and three named sections.  
**Before:** LLM assembles Markdown by reading the format spec.  
**After:** `extracted/render.py:render_report` — a pure string formatter that takes typed dicts and produces the exact Markdown structure the skill prescribes.

---

### Extraction 9 — Parallel data pulls (Pattern 8)
**Source:** `SKILL.md:18–32` — four data sources listed sequentially  
**Before:** Agent executes four tool calls one at a time (4–8 turns).  
**After:** `entry_nodes` lists all four fetch nodes; adapter runs them in the same execution wave.

---

## Phase 4: LLM-Judge Extraction

**None.** Every decision in this skill is numeric threshold comparison or keyword detection — no step requires reading prose, job titles, or free-text to make a classification. The skill has no rubric-driven classification. All 12 non-HITL nodes are either `external_call` or `pure_function`.

---

## Phase 6: Adapter Checklist

- [x] Every `pure_function` / `external_call` node has an `impl:` pointing at a created file
- [x] All four MCP data-pull nodes carry `mcp:` bindings (`server` + `tool` + `args`)
- [x] All `pure_function` nodes are fully implemented (no `NotImplementedError` stubs)
- [x] `external_call` stubs raise `NotImplementedError` with the API description (Regime 1 — no probe-context)
- [x] No `llm_judge` nodes → no `signature:` requirements
- [x] No `agent_loop` nodes → no `tools:` requirements
- [x] `hitl_gate` has `signal: manual_data_provided`
- [x] All edge `from`/`to` reference real node IDs
- [x] `entry_nodes` and `exit_nodes` reference real node IDs
- [x] No `mandatory: true` on agent_loop nodes (no agent_loop nodes exist)
- [x] No YAML key named `on:` (used `on_signal:` throughout)
- [x] All `inputs:` references use valid grammar forms and reference upstream nodes
- [x] Every implemented `pure_function` is covered by a test file under `tests/`
- [x] Every `# skill: "..."` citation quotes text that appears in `SKILL.md`
- [x] Every `# rote: assumption:` has a corresponding Open question below

---

## Open questions

### OQ-1: Google Drive MCP tab/sheet selection
**The skill says:** `"Tab: Dwell Log - Form Responses"` for the dwell ticket file.  
**What we chose:** We set `constants.tab: "Dwell Log - Form Responses"` but only mapped `file_id` in `mcp.args` because it is unknown whether the `google_drive.read_file_content` tool accepts a sheet/tab name argument.  
**Action needed:** Verify the MCP tool's argument list. If `read_file_content` accepts a `sheet_name` or `tab` argument, add it to `fetch_dwell_tickets.mcp.args: {file_id: file_id, sheet_name: tab}`. If not, add a `pure_function` node after the fetch to filter the raw content to the correct tab before passing it to `parse_dwell_tickets`.

---

### OQ-2: Gmail `search_threads` result shape
**The skill says:** use `gmail.search_threads` with query `label:dock-activity newer_than:1d`.  
**What we chose:** `parse_dock_activity_emails` assumes the result is `{"threads": [{subject: ..., snippet: ...}]}`. Appointment status is inferred by scanning thread subjects/snippets for the words "approved" / "request".  
**What the skill does not state:** the actual email subject format. If dock-activity emails use a different scheme (e.g., a status field, a structured subject line, or a separate label per status), the keyword heuristic may over- or under-count.  
**Action needed:** After first production run, compare `approved_count + requested_count` against `total_threads`. If many threads are uncategorized, inspect the actual subjects and update the keyword list or add a classification step.

---

### OQ-3: Dwell threshold dimension ("Brand" vs "Carrier")
**The skill says:** "Brand 25-40 = Missort Warning, 41+ = Missort Alert. Carrier 80+ missing = Possible Misload Alert."  
**What we chose:** Both `parse_dwell_tickets` and `apply_dwell_thresholds` apply a single tiered threshold to the row's `packages` column: 80+ = Possible Misload Alert (takes precedence), 41–79 = Missort Alert, 25–40 = Missort Warning. We did not aggregate by carrier name because the ticket log and BI dashboard data each have only one package count per row, not separate brand-count and carrier-count columns.  
**`# rote: assumption:`** in `extracted/parsers.py:_classify_dwell_packages` and `extracted/thresholds.py:_classify_dwell_row`.  
**Action needed:** If the BI dashboard or ticket log has separate "brand packages" and "carrier packages" columns, rewrite both classifiers to use the appropriate column for each threshold. The current implementation would then need to be split into a brand check and a carrier check.

---

### OQ-3a: Assumption — `_classify_dwell_packages` per-row interpretation (`extracted/parsers.py`)
**Assumption text:** `"carrier >= 80 (Possible Misload Alert)" is applied to the raw # packages column of each open ticket row … The skill does not state whether this is per-row or aggregated by carrier name; per-row is chosen as the simpler/more actionable interpretation.`  
**Skill gap:** The skill says "carrier >= 80 (Possible Misload Alert)" but does not say whether to apply the threshold per-row or to aggregate packages across all rows for each carrier before comparing. Per-row is assumed.  
**Action needed:** Confirm with ops team whether the carrier alert should aggregate all dwelling packages for a carrier across sites before applying the 80-package threshold.

---

### OQ-3b: Assumption — `_classify_dwell_packages` range priority (`extracted/parsers.py`)
**Assumption text:** `thresholds are applied as ranges (not cumulative), with the highest threshold taking precedence: 80+ = Possible Misload Alert, 41-79 = Missort Alert, 25-40 = Missort Warning.`  
**Skill gap:** The skill lists three thresholds (`>= 25`, `>= 41`, `>= 80`) without stating that they are mutually exclusive or that the highest wins. A row with 80 packages meets all three; the code returns only "Possible Misload Alert". If all three alerts should fire simultaneously, the implementation must return a list rather than a single string.  
**Action needed:** Confirm whether a row with 80+ packages should show one alert (highest only) or all three.

---

### OQ-3c: Assumption — Gmail `search_threads` result shape (`extracted/parsers.py`)
**Assumption text:** `The Gmail MCP search_threads tool returns a JSON-serializable dict with a 'threads' key. Each thread has a 'subject' or 'snippet' field. Appointment status is inferred from the word 'approved' (→ Approved) or 'request'/'requested' (→ Requested) appearing in the thread subject/snippet.`  
**Skill gap:** The skill specifies the Gmail search query but not the result structure or how "Approved" vs "Requested" is signalled in the emails. The keyword heuristic is the simplest defensible default; it may mis-classify if subject lines use different wording.  
**Action needed:** (Same as OQ-2 — listed here to map to the specific `# rote: assumption:` marker in the code.)

---

### OQ-3d: Assumption — `_classify_dwell_row` tiered threshold (`extracted/thresholds.py`)
**Assumption text:** `"Brand 25-40 = Missort Warning, 41+ = Missort Alert, Carrier 80+ missing = Possible Misload Alert" is applied as a single tiered threshold on the packages column (highest threshold wins). The skill names Brand/Carrier as separate dimensions but does not provide separate package counts per dimension; a single per-row packages column is used.`  
**Skill gap:** Identical to OQ-3 and OQ-3a but appearing in `thresholds.py:_classify_dwell_row` (applied to BI dashboard data) rather than `parsers.py:_classify_dwell_packages` (applied to ticket log data). Both functions share the same assumption; both need the same confirmation.  
**Action needed:** (Same as OQ-3a — listed here to map to the specific `# rote: assumption:` marker in `extracted/thresholds.py`.)

---

### OQ-4: Sorter metrics availability for non-automated sites
**The skill says:** "for sites with automated divert lines" — implying some sites may have no sorter data.  
**What we chose:** `apply_sorter_flags` accepts an empty `sorter_metrics` list and returns empty site lists (no flags), so sites without sorters simply produce no output in Section 2.  
**Action needed:** Confirm with the duty manager team whether sites without sorters are expected to send `sorter_metrics: []` or to omit the key entirely. The current HITL gate output schema accepts an empty list; a missing key would cause a KeyError in `apply_sorter_flags` unless the gate's signal handler normalizes missing keys to `[]`.

---

### OQ-5: File IDs are example values
**The skill uses:** `1ExampleShipmentContainersFileId00000000000`, `1ExampleDockPendingLogFileId0000000000000000`, `1ExampleDwellTicketLogFileId0000000000000000`.  
These are clearly placeholder IDs. Before deploying, replace all three `constants.file_id` values in `pipeline.yaml` with the real production Google Drive file IDs. The skill does not provide them.

---

## Suggested Next Steps

1. **Replace placeholder file IDs** (OQ-5) and run the pipeline once manually to verify the four fetch nodes return parseable content before scheduling.

2. **Verify Gmail thread shape** (OQ-2) by logging the first `search_threads` response in a dev run, then adjusting `parse_dock_activity_emails` if the subject/snippet heuristic is inaccurate.

3. **Confirm Drive tab support** (OQ-1) by checking the MCP tool's declared arguments or testing a call with a `sheet_name` parameter.

4. **Run `python -m pytest tests/`** from the work directory — all tests are offline and should pass against the current implementations. Any failure points to a mis-stated skill rule.

5. **First production run dogfood:** after the first real run, compare `render_report` output against the existing human-authored report to validate the template matches expectations. The Section 3 dwell threshold interpretation (OQ-3) is the most likely source of discrepancy.

6. **DSPy/BAML compile not needed** — this skill has zero `llm_judge` nodes. The compiled pipeline never calls an LLM at runtime. Skip the signature-optimization phase entirely.

7. **Schedule tuning:** the pipeline is set to `0 6 * * *` (06:00 daily). Adjust to match the duty manager's shift start. The HITL gate timeout is 4 h; if the duty manager needs until end-of-shift, increase to `8h` or `10h`.
