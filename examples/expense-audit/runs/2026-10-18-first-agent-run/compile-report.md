# Compilation report: expense-audit

Source skill: `src/examples/expense-audit/skill/SKILL.md`
Compiled: 2026-10-18

---

## Summary metrics

| Metric | Value |
| --- | --- |
| Total nodes | 8 |
| `pure_function` | 6 |
| `external_call` | 1 |
| `hitl_gate` | 1 |
| `llm_judge` | 0 |
| `agent_loop` | 0 |
| Mandatory nodes | 7 (all non-HITL) |
| HITL gates | 1 |
| Entry nodes (parallel) | 3 |

**Codifiability:** 7 of 7 non-HITL nodes are deterministic code (`pure_function` or `external_call`). That is 100% of the codifiable surface. The expense audit has no fuzzy classification. Every rule is a boolean or numeric comparison against a fixed threshold.

**Estimated token reduction:** An agent run of this skill takes an estimated 13–95 turns depending on batch size (see `eval.yaml`). The compiled pipeline makes zero LLM calls at run time. All inference cost is eliminated.

**HITL gate:**

| Gate | Blocks on | Timeout |
| --- | --- | --- |
| `controller_review_gate` | Controller approval/rejection of REVIEW-severity findings | 3 days |

---

## Node classification table

| Step | Node ID | Kind | Justification |
| --- | --- | --- | --- |
| Step 1 | `convert_to_usd` | `pure_function` | Deterministic multiplication against a fixed rate table embedded as constants. |
| Step 2 | `check_duplicates` | `pure_function` | Tuple-equality comparison with fixed key fields. No judgment involved. MANDATORY. |
| Step 3 | `check_late_submissions` | `pure_function` | Date arithmetic against a fixed 60-day window. MANDATORY. |
| Step 4 | `apply_policy_rules` | `pure_function` | All rules are boolean/numeric comparisons (cap > amount, amount >= threshold, field presence). No prose interpretation required. |
| Step 5 | `build_review_queue` | `pure_function` | Fixed aggregation and sort. Pure formatting from already-structured inputs. |
| Step 6 | `controller_review_gate` | `hitl_gate` | Skill explicitly says "Post the review queue to the controller and wait for a decision." |
| Step 7a | `render_audit_report` | `pure_function` | Fixed template from `references/report-format.md`. Fills in values from structured inputs. |
| Step 7b | `save_audit_report` | `external_call` | Deterministic file write. Needs retry/timeout semantics for a network path. |

---

## Crystallization log

Every extraction below moves a procedure from English prose into Python code, making it testable, pinned, and impossible to silently skip.

### 1. `MONTH_END_RATES` constant dict: `extracted/fx.py`

**Source:** `references/fx-rates.md`, lines 9–26

**Before (prose):** The agent reads a markdown table at runtime and mentally applies the rates.

**After (code):**
```python
MONTH_END_RATES: dict[str, dict[str, float]] = {
    "2026-09": {"USD": 1.000000, "EUR": 1.105000, "GBP": 1.312500, "CAD": 0.724000, "JPY": 0.006780},
    "2026-08": {"USD": 1.000000, "EUR": 1.098000, "GBP": 1.301000, "CAD": 0.731000, "JPY": 0.006850},
}
```

**Constants lifted:** 10 rate values, 2 period keys.

---

### 2. MANDATORY duplicate check: `extracted/checks.py:check_duplicates`

**Source:** `SKILL.md`, Step 2

**Before (prose):** "Two lines are duplicates when they have the same employee, merchant (ignore case), amount, currency, and date. Keep the first one submitted (lowest expense id)… This check is mandatory. Run it on every line, every month…"

**After (code):** `check_duplicates()` groups by `(employee, merchant.lower(), amount, currency, date)`, sorts each group by `expense_id`, and flags all but the first as BLOCK findings. `mandatory: true` on the IR node means the workflow cannot skip it.

**Risk removed:** In a long agent trajectory, prompt drift or model upgrade could cause the agent to skip this check on "obviously clean" batches. The compiled pipeline calls this function unconditionally on every run.

---

### 3. MANDATORY late submission check: `extracted/checks.py:check_late_submissions`

**Source:** `SKILL.md` Step 3 and `references/expense-policy.md` line 28

**Before (prose):** "Flag any line dated more than 60 days before the last day of the reporting period…"

**After (code):**
```python
LATE_SUBMISSION_DAYS: int = 60
```
`check_late_submissions()` computes `last_day` of the period, then flags lines with `date < (last_day - 60 days)` or `date > last_day`.

**Constants lifted:** `LATE_SUBMISSION_DAYS = 60`

---

### 4. Policy constants: `extracted/policy.py`

**Source:** `references/expense-policy.md`, lines 4–48

**Before (prose):** Agent reads the policy table at runtime to determine caps and thresholds.

**After (code):**
```python
RECEIPT_REQUIRED_THRESHOLD: float = 25.00
REPORT_TOTAL_LIMIT: float = 2500.00
CATEGORY_CAPS: dict[str, float | None] = {
    "airfare": 900.00,
    "lodging": 325.00,   # per night
    "meals": 75.00,
    "ground_transport": 150.00,
    "software": 100.00,
    "office_supplies": 250.00,
    "entertainment": None,
}
```

**Constants lifted:** 8 cap values, 2 threshold values, 1 set of known categories.

**Note on lodging:** The cap is `$325.00 × line.nights`. The `nights` multiplier is policy behavior, not a constant to lift. The implementation must apply the multiplication.

---

### 5. Fixed report template: `extracted/report.py`

**Source:** `references/report-format.md`, lines 6–38

**Before (prose):** Agent reads the format file and reconstructs the markdown tables by hand every run, occasionally mis-formatting column order or decision column values.

**After (code):** `_REPORT_TEMPLATE` is a module-level constant with Python `.format()` placeholders. `render_audit_report()` fills it from typed inputs. The format cannot drift.

**Decision column rules also codified:**
- BLOCK rows: always `"returned"` (the controller cannot override this).
- REVIEW rows: `decisions.get(expense_id, "pending")`.

---

### 6. Parallel entry wave (Pattern 8)

**Source:** `SKILL.md`, introduction to steps 1–3: "Steps 1 to 3 do not depend on each other, so do them together."

**Before:** An agent executes steps 1, 2, 3 sequentially (one tool call at a time).

**After:** Three nodes in the same execution wave (`entry_nodes: [convert_to_usd, check_duplicates, check_late_submissions]`). Adapters run same-wave nodes concurrently. This is the primary wall-clock speedup from compilation: for a 50-line batch the three checks run in ~max(t1, t2, t3) instead of ~(t1 + t2 + t3).

---

## Phase 4: LLM-judge extraction

**No `llm_judge` nodes.** Every decision in this skill is a deterministic boolean or numeric comparison. The policy table maps to `if/else` chains, not a rubric requiring LLM judgment.

This is unusual among compiled skills. It means the token savings are 100% (from the non-HITL steps). The only LLM involved in a compiled run is any model the operator uses to drive the controller review interaction itself, which is out of scope.

---

## Open questions for human review

### OQ-1: Unknown FX period handling

**What the skill says:** "If a line uses a currency that isn't in the table, stop and ask finance for the rate. Do not look one up online."

**Decision made:** `convert_to_usd()` raises `RuntimeError` when the period or currency is missing. This parks the workflow (durable runtimes retry or alert on uncaught exceptions).

**Verify:** Finance should confirm the preferred escalation path: either supply `fx_rates` in the pipeline input before each run, or handle the `RuntimeError` with an alerting `on_failure` hook.

---

### OQ-2: Controller signal validation

**What the skill says:** "If the controller names an expense id that isn't in the queue, ask again rather than guessing what they meant."

**Decision made:** This validation is not modeled in the IR. The HITL gate accepts any `decisions` payload the controller sends. The runtime would need to implement signal-side validation (check the submitted `expense_id` values against `review_queue`) before releasing the gate.

**Verify:** This may require a custom signal handler in the adapter that validates the payload before calling `signal.send()`. It is currently an open gap. The compiled pipeline will silently accept a bad `expense_id` and produce a report with a decision on a phantom line.

---

### OQ-3: Finance shared folder path

**What the skill says:** "save it to the finance shared folder as `<period>-expense-audit.md`"

**Decision made:** The folder path is a pipeline input field (`finance_folder_path`). This requires the caller to supply the path on each run. Passing `null` skips the save (dry-run mode).

**Verify:** If the finance team uses a cloud storage path (e.g., Google Drive, SharePoint, S3) rather than a network filesystem path, the `save_audit_report` stub will need a different implementation, likely an `mcp:` binding to a cloud storage tool.

---

### OQ-4: Escalation on HITL timeout

**What the skill says:** "Hold the audit open for up to three days before escalating to the finance manager."

**Decision made:** The `controller_review_gate` has `timeout: 3d`. When it fires, the pipeline fails with a timeout error. The `on_failure: notify_owner` config hook is where the escalation notification would go.

**Verify:** Whether "escalate to the finance manager" means notify only, or re-route the gate to a different approver, is not specified. If re-routing is needed, a second HITL gate node should be added.

---

### OQ-5: Notify template runtime resolution

**What the skill says:** Step 6 posts the review queue to the controller.

**Decision made:** The `controller_review_gate` notify block has `target: "#finance-expense-review"`. This is a placeholder Slack channel. The message template references `{period}` which must be resolved at runtime from the pipeline input.

**Verify:** Confirm the correct Slack channel name. The `build_review_queue.output.review_queue` data is also available at the gate but is not threaded into the notify message. Confirm whether the controller receives the queue inline or via a separate link.

---

## Suggested next steps

1. **Implement the stubs.** All five `extracted/*.py` files raise `NotImplementedError`. Fill them in, starting with `checks.py` (pure date and tuple logic, easiest to unit-test). Add `tests/test_extracted.py` with edge cases: empty batch, single duplicate pair, lodging with `nights=None`, period boundary for late submissions.

2. **Run `rote emit pipeline.yaml --runtime dbos --out /tmp/expense-audit-dbos`** to produce the durable Python workflow. The IR validated cleanly. Emission should succeed immediately.

3. **Resolve OQ-2 (signal validation).** The phantom-expense-id gap is the most likely correctness issue in a production run. Consider adding a `pure_function` node between `controller_review_gate` and `render_audit_report` that validates the decision keys against the review queue.

4. **Resolve OQ-3 (save target).** If the finance team uses cloud storage, replace `save_audit_report` with an `external_call` that has an `mcp:` binding to the appropriate storage tool.

5. **Seed evals.** Even though there are no `llm_judge` nodes, writing golden-fixture tests for the policy rules (e.g., a lodging line at exactly $325.00 × nights vs. $325.01 × nights) will catch regressions when policy thresholds change.
