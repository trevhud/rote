# Graduation Report — invoice-push

**Source skill:** `examples/invoice-push/skill/SKILL.md`
**Graduated:** 2026-07-18
**Default runtime:** DBOS (emit with `rote emit pipeline.yaml --runtime dbos`)

---

## Summary Metrics

| Metric | Value |
|---|---|
| Total nodes | 15 |
| `pure_function` | 4 (27%) |
| `external_call` | 10 (67%) |
| `agent_loop` | 1 (6%) |
| `llm_judge` | 0 |
| `hitl_gate` | 0 |
| **Codifiable (pf + ec)** | **14 / 15 = 93%** |
| MANDATORY nodes | 3 (`check_prerequisites`, `clear_filters`, `check_row_eligibility`) |

**Estimated token reduction per run:** 85–90%. This is a browser-automation
loop skill with zero fuzzy classification. The pre-graduation cost is dominated
by the per-row agent loop (6–10 turns × row count = potentially hundreds of
turns). After graduation, only `process_invoices_loop` remains agentic; its
four loop-body sub-nodes are deterministic code invoked by the orchestrated
loop. The LLM's only job post-graduation is coordinating the browser cursor
across pages.

**HITL gates:** None. The pipeline is fully automated from trigger to report.

**Eval sidecar:** `eval.yaml` — estimated 30–680 turns pre-graduation depending
on qualifying row count. Typical run (~22 invoices) ≈ 140–240 turns.

---

## Crystallization Log

Every prose-to-code extraction from Phase 3:

### 1 — Date window formula (SKILL.md:55–58)

**Before (prose):**
> Calculate dynamically before opening the browser:
> - End date (yesterday): today − 1 day
> - Start date: end date − 7 days (7-day rolling window)
> - Format: M/D/YYYY (e.g., 4/22/2026)

**After (code):** `extracted/date_utils.py:calculate_date_window()`
```python
WINDOW_DAYS: int = 7

def calculate_date_window(override_date=None) -> DateWindow:
    end = today - timedelta(days=1)
    start = end - timedelta(days=WINDOW_DAYS)
    return DateWindow(start_date=_format_date_mdy(start), end_date=_format_date_mdy(end))
```
Constants lifted: `WINDOW_DAYS = 7`, `DATE_FORMAT = "M/D/YYYY"`.

---

### 2 — 24-hour rule (SKILL.md:65, 161–165) — MANDATORY

**Before (prose):**
> Invoices imported today are **never** pushed.
> If **Date Imported** is within 24 hours of the current run time: Do **not** push.
> Log as: ⚠️ Skipped | Failure Detail: `Imported within 24 hours of run — not eligible.`

**After (code):** `extracted/eligibility.py:check_row_eligibility()` — second rule:
```python
RECENT_HOURS: int = 24
if age < timedelta(hours=RECENT_HOURS):
    return EligibilityResult(eligible=False, outcome=SKIP_RECENT, ...)
```
Node `check_row_eligibility` has `mandatory: true` — cannot be skipped.

---

### 3 — Status eligibility table (SKILL.md:151–158) — MANDATORY

**Before (prose):**
> | Status | Action |
> | IMPORTED | ✅ Eligible — proceed to 7c |
> | FAILED | ⚠️ Skip — log: Status is FAILED — not eligible for push. |
> | IN PROGRESS | ⚠️ Skip — ... |
> ⚠️ Never click ⋮ on a FAILED or IN PROGRESS row.

**After (code):** `extracted/eligibility.py:check_row_eligibility()` — first rule:
```python
ELIGIBLE_STATUS: str = "IMPORTED"
if status != ELIGIBLE_STATUS:
    return EligibilityResult(eligible=False, outcome=SKIP_STATUS,
                             skip_reason=f"Status is {status} — not eligible for push.")
```
Node `check_row_eligibility` has `mandatory: true`. The adapter emits this as
an unconditional activity that always runs before `push_invoice`.

---

### 4 — Page size constant (SKILL.md:105)

**Before (prose):** Click the "25 per page" dropdown → select **90**.

**After (code):** `extracted/browser_ops.py`:
```python
PAGE_SIZE: int = 90
```
Also lifted into `pipeline.yaml` `constants.page_size: 90` on `set_page_size`.

---

### 5 — Toast timeout constant (SKILL.md:181)

**Before (prose):** Wait for Toast Notification (up to 5 seconds)

**After (code):** `extracted/browser_ops.py`:
```python
TOAST_TIMEOUT_SECONDS: int = 5
```

---

### 6 — Toast → error code fixed mapping (SKILL.md:183–191)

**Before (prose):** A table of toast messages, colors, and result codes.

**After (code):** `extracted/browser_ops.py`:
```python
TOAST_TO_ERROR_CODE: dict[str, str] = {
    "Procurement is not enabled for this invoice type": "ERR-INVOICE-TYPE",
    "Invoice already exists in procurement": "ERR-DUPLICATE",
    "Procurement request timed out": "ERR-TIMEOUT",
    "Authorization denied": "ERR-AUTH",
    "Cannot connect to procurement": "ERR-CONN",
}
FATAL_ERROR_CODES: frozenset[str] = frozenset({"ERR-AUTH", "ERR-CONN"})
```
`push_invoice` raises `FatalPushError` (a typed exception) for fatal codes — the
durable executor treats this as a non-retriable workflow failure, surfaces the
batch_id of the last attempted row, and prevents further processing.

---

### 7 — Stale filter MANDATORY guard (SKILL.md:86)

**Before (prose):**
> ⚠️ Stale filters silently exclude qualifying invoices. Always start clean.

**After (code):** Node `clear_filters` has `mandatory: true`. The adapter emits
this as an unconditional step that cannot be skipped or made conditional,
regardless of observed filter state.

---

### 8 — Prerequisites MANDATORY guard (SKILL.md:49)

**Before (prose):**
> Do not proceed until both prerequisites are confirmed.

**After (code):** Node `check_prerequisites` has `mandatory: true`. Raises
`RuntimeError` with an operator-facing message; the workflow fails rather than
silently proceeding without a browser connection.

---

### 9 — Archive folder ID constant (SKILL.md:214)

**Before (prose):**
> **Archive folder ID:** `1ExampleInvoicePushArchiveFolderId0000000000`

**After (code):** `extracted/drive_ops.py`:
```python
ARCHIVE_FOLDER_ID: str = "1ExampleInvoicePushArchiveFolderId0000000000"
REPORT_FILE_PREFIX: str = "Invoice_Push_Report"
```
Also lifted into `pipeline.yaml` `constants` on `save_report_to_drive`.

---

### 10 — Fixed 18-column Invoice Detail tab schema (SKILL.md:225–247)

**Before (prose):** Numbered table of 18 columns with Source column.

**After (code):** `extracted/report_builder.py`:
```python
INVOICE_DETAIL_COLUMNS: list[str] = [
    "Batch ID", "Status", "Invoice Number", "Carrier", "Invoice Type",
    "Import File", "Date Imported", "Imported By", "Record Count", "Charges",
    "Packages", "Ship Packages", "Earliest Ship Date", "Latest Ship Date",
    "Sent Date", "Push Status", "Failure Code", "Failure Detail",
]
```
`build_report()` assembles rows in this exact order with typed inputs.

---

### 11 — Static Failure Code Reference table (SKILL.md:263–268)

**Before (prose):** "Static reference table, one row per code from §7f: code,
toast message, meaning, and the operator action."

**After (code):** `extracted/report_builder.py`:
```python
FAILURE_CODE_REFERENCE: list[dict[str, str]] = [
    {"Code": "ERR-INVOICE-TYPE", "Toast Message": "...", "Meaning": "...", "Operator Action": "..."},
    # ... one entry per code
]
```
Tab 3 is generated from this constant — the LLM never re-derives it.

---

### 12 — Run Summary 7-metric schema (SKILL.md:250–261)

**Before (prose):** Table of 7 metric / value rows.

**After (code):** `build_report()` in `extracted/report_builder.py` produces
`run_summary_rows` as a typed list of `{"Metric": ..., "Value": ...}` dicts,
with per-ERR-code breakdown computed arithmetically from `InvoiceResult` objects.

---

## Open Questions

### OQ-1 — Browser automation MCP server identity

The skill references a "browser-automation plugin" by UI location (Settings →
Plugins) but does not name the underlying MCP server. The `process_invoices_loop`
`tools:` list uses generic names (`browser_navigate`, `browser_click`, etc.).

**What I chose:** generic tool names in the IR; the adapter emits stubs.

**What to verify:** Replace the `tools:` list in `process_invoices_loop` with the
actual MCP tool names from your browser plugin, and add `mcp:` bindings to the
10 browser `external_call` nodes when wiring for production. The stubs in
`extracted/browser_ops.py` document which browser operation each function wraps.

---

### OQ-2 — Google Drive / Sheets MCP server identity

The skill uses Google Drive for the report but does not name an MCP server.
`save_report_to_drive` is stubbed with `NotImplementedError`.

**What to verify:** If you have a Google Drive MCP server available, add an
`mcp:` binding to `save_report_to_drive` and implement the stub by calling the
Drive + Sheets API tools.

---

### OQ-3 — `check_prerequisites` error messaging

The skill specifies an exact error message to display:
> "I cannot run the invoice push — the browser-automation plugin is not
> connected..."

**What I chose:** `check_prerequisites` raises `RuntimeError` with an
operator-facing message. In the graduated pipeline this becomes a workflow
failure; the runtime surfaces the exception message.

**What to verify:** If the workflow needs to return a user-visible message
rather than fail, add a `hitl_gate` or a terminal `pure_function` that formats
the error before surfacing it. For the current requirement ("do not proceed"),
the `RuntimeError` approach is correct.

---

### OQ-4 — Date Imported timezone parsing

`read_invoice_row` must parse `date_imported_dt` (datetime) from the cell's raw
text for the 24-hour rule. The skill shows "Full timestamp with timezone" but
does not specify the format (e.g., `5/4/2026 9:15 AM MDT`).

**What to verify:** Inspect the actual platform timestamp format and implement
the parser in `read_invoice_row`. A 1-hour offset from DST ambiguity won't
matter in practice given the 24-hour threshold, but the timezone should be
parsed correctly to avoid off-by-one-day edge cases near midnight.

---

### OQ-5 — `process_invoices_loop` stopping on FatalPushError

The skill says ERR-AUTH and ERR-CONN should "stop run; alert user." In the
graduated pipeline, `push_invoice` raises `FatalPushError` which terminates
the `agent_loop` and marks the workflow as failed.

**What to verify:** On the target runtime (DBOS/Temporal), confirm the
exception bubbles out of the loop activity and halts the workflow rather than
being swallowed. DBOS will mark the workflow failed and the exception message
(including `batch_id`) will appear in the workflow history.

---

### OQ-6 — Pagination after ERR-TIMEOUT and ERR-NO-CONFIRM

For ERR-TIMEOUT and ERR-NO-CONFIRM, the skill says "check the portal manually"
but does not say to stop the run. The loop should continue to the next row.

**What I chose:** These are non-fatal — `push_invoice` returns a `PushResult`
with `is_success=False` and the error code, and the loop continues.

**What to verify:** Confirm this matches the operator's expectations. If ERR-TIMEOUT
after N consecutive rows should stop the run, add a counter to `process_invoices_loop`.

---

## Suggested Next Steps

1. **Implement `check_prerequisites`** — this is the first blocker for any real
   run. It just needs to verify the browser plugin and URL; the rest of the pipeline
   can be tested without it.

2. **Wire the browser MCP bindings** — add `mcp:` bindings to the 10 browser
   `external_call` nodes pointing at your browser automation server, then implement
   each stub in `extracted/browser_ops.py`.

3. **Implement `save_report_to_drive`** — the final output. Can be tested in
   isolation with a synthetic `RunReport`.

4. **Run `rote eval` against the BDR-scale skill** — use `--run` mode with a
   small set of qualifying invoices (5–10 rows) to empirically measure the
   before/after cost and verify the 24-hour rule and toast classification work
   correctly.

5. **Dogfood `check_row_eligibility` first** — it has zero external dependencies
   and covers the two MANDATORY business rules. Write pytest tests from the
   eligibility table in `SKILL.md §7b–7c` before running any live browser sessions.

6. **Consider re-graduating after first production run** — if the browser DOM
   structure requires non-trivial navigation logic (e.g., the ⋮ menu appears in a
   shadow DOM), the `push_invoice` external_call stub may warrant richer
   documentation. The IR is stable; only the implementation stubs need updating.
