# Compilation Report — deal-monitor

**Source skill:** `examples/deal-monitor/skill/SKILL.md`
**Compiled:** 2026-07-18

---

## Summary Metrics

| Metric | Value |
|---|---|
| Total nodes | 12 |
| `pure_function` | 4 |
| `external_call` | 4 |
| `llm_judge` | 4 |
| `agent_loop` | 0 |
| `hitl_gate` | 0 |
| Mandatory nodes | 1 (`apply_inclusion_filter`) |
| HITL gates | None |
| % codifiable (non-HITL) | 67% (8 of 12) |
| Agent turn estimate before | 50–260 turns/run |
| Extracted Python modules | 5 (`slack.py`, `gmail.py`, `filters.py`, `matching.py`, `dashboard.py`) |
| LLM-judge signatures | 4 (`extract_deal_fields`, `extract_thread_data`, `classify_thread_status`, `score_new_opportunity`) |
| Eval seed files | 4 (17 total examples) |

The 67% codifiable rate is right at the midpoint of a well-compiled skill (target 60–70%). The
4 deterministic nodes do the heavy structural work (fetch, filter, match, render); the 4 LLM
judges handle the unbounded-input steps that genuinely need reading comprehension.

**No HITL gates.** This skill produces a daily artifact (HTML file + chat summary) and never
needs human approval to proceed. The output lands in `./outputs/deal-monitor.html`; the operator
reviews it by opening the file, not by sending a signal back to the workflow.

---

## Crystallization Log

Every prose-to-code extraction from Phase 3, with source location and before/after.

### C-1: `SLACK_CHANNEL_ID` and `SLACK_MESSAGE_LIMIT`

**Source:** SKILL.md line 22
**Prose:** "Read the 50 most recent messages from #deal-intake (channel ID: `C0EXAMPLE000`)"
**Extracted to:** `extracted/slack.py`

```python
# Before (runtime re-derivation):
# The agent reads "channel ID: C0EXAMPLE000, 50 messages" on every run

# After (codified):
SLACK_CHANNEL_ID = "C0EXAMPLE000"
SLACK_MESSAGE_LIMIT = 50
```

These are constants that will never change run-over-run. Codifying them prevents a model
drift event from using the wrong channel ID.

---

### C-2: EU/UK inclusion filter (MANDATORY)

**Source:** SKILL.md lines 26–29
**Prose:**
> Include: EU/UK: ALL opps — any UK, EU, or non-US location, plus anything submitted by Alex
> Rivers or Sam Patel (always UK). North America: Only if SKU count < 2,000. Unknown SKU =
> include, flag as risk. Confirmed ≥ 2,000 = exclude.

**Extracted to:** `extracted/filters.py:apply_inclusion_filter`
**Marked `mandatory: true` in IR.**

```python
# Before (agent judgment per run):
# "Apply these rules..." — model might skip, misread, or misclassify

# After (codified):
MAX_NA_SKU_COUNT = 2000
ALWAYS_UK_SUBMITTERS = frozenset({"Alex Rivers", "Sam Patel"})

def apply_inclusion_filter(deals) -> FilterResult:
    # deterministic: location check + submitter check + SKU threshold
```

**Why this matters most:** This is a business rule with financial implications — including a
2,500-SKU NA opp wastes quoting effort. Making it `mandatory: true` means the IR validator and
every adapter emit it as an unconditional step that cannot be skipped.

---

### C-3: `GMAIL_SEARCH_DAYS`

**Source:** SKILL.md line 38
**Prose:** "Use the last 60 days as the window."
**Extracted to:** `extracted/gmail.py`

```python
GMAIL_SEARCH_DAYS = 60
```

---

### C-4: Gmail fixed subject patterns

**Source:** SKILL.md lines 33–37
**Prose:** Five specific search patterns enumerated as an explicit list.
**Extracted to:** `extracted/gmail.py`

```python
GMAIL_FIXED_SUBJECT_PATTERNS = ["Quote Request", "New Business", "RFP"]
```

The 4th search (warehouse domains) is a known pattern; the 5th (account names) is dynamically
derived from deal records. `build_gmail_queries()` assembles all 5 types so the agent never
has to re-read the search spec.

---

### C-5: Gmail thread URL template

**Source:** SKILL.md line 38
**Prose:** "format: `https://mail.google.com/mail/u/0/#inbox/[threadId]`"
**Extracted to:** `extracted/gmail.py:build_thread_url`

```python
GMAIL_THREAD_URL_TEMPLATE = "https://mail.google.com/mail/u/0/#inbox/{thread_id}"
```

Without this codification, every run risks the model formatting the URL slightly differently
(with `#search/`, without `u/0/`, etc.), breaking the links in the dashboard.

---

### C-6: `PipelineStep` enum and `NEEDS_ATTENTION_STEPS`

**Source:** SKILL.md line 43
**Prose:** "Steps: 1=RFP Sent, 2=Response Needed, 3=Follow-Up Needed, 4=Pricing Returned,
5=Pricing Sent, 6=Declined."
**Extracted to:** `signatures/classify_thread_status.py`

```python
class PipelineStep(IntEnum):
    RFP_SENT = 1
    RESPONSE_NEEDED = 2
    FOLLOW_UP_NEEDED = 3
    PRICING_RETURNED = 4
    PRICING_SENT = 5
    DECLINED = 6

NEEDS_ATTENTION_STEPS = frozenset({PipelineStep.RESPONSE_NEEDED, PipelineStep.FOLLOW_UP_NEEDED})
```

The dashboard's color-coding is derived from `NEEDS_ATTENTION_STEPS` in `extracted/dashboard.py`,
ensuring the "needs attention" logic stays in sync with the classification rubric.

---

### C-7: Dashboard HTML template

**Source:** SKILL.md lines 49–53
**Prose:** "Generate a self-contained HTML file with three tabs: New Opportunities | Quoting |
Closed. Use a clean, self-contained layout — no external assets — with a summary tile row above
the tabs and color-coded needs-attention rows."
**Extracted to:** `extracted/dashboard.py:generate_dashboard_html`

The entire HTML structure is codified as a Python function. The "color-coded needs-attention rows"
is implemented deterministically based on `NEEDS_ATTENTION_STEPS` — no LLM required to decide
which rows to highlight.

---

### C-8: Summary format

**Source:** SKILL.md line 54
**Prose:** "new opps count, active quoting count (with how many need attention), closed count"
**Extracted to:** `extracted/dashboard.py:generate_summary`

```python
# After:
f"**Deal Monitor** — {len(new_opps)} new opp{'s' if ...}"
f" · {len(active)} active quoting ({len(needs_attention)} need attention)"
f" · {len(closed)} closed"
```

The format is fixed. No LLM needed to count or format this string.

---

### C-9: Fuzzy match threshold

**Source:** SKILL.md line 38 ("fuzzy match fine")
**Prose:** Implied threshold for accepting an account name match.
**Extracted to:** `extracted/matching.py`

```python
FUZZY_MATCH_THRESHOLD = 80  # out of 100 (rapidfuzz partial_ratio)
```

The choice of 80 is a reasonable default (see Open Questions Q-2).

---

## Open Questions

These are judgment calls made during compilation that the human reviewer should verify.

### Q-1 (HIGH): deal-scoring rubric is missing — `score_new_opportunity` is a placeholder

**What I found:** SKILL.md Step 4 reads: "Read the deal-scoring skill at
`~/.claude/skills/deal-scoring/SKILL.md` and apply its scoring rules." The file does not exist
in this workspace.

**What I decided:** Classified `score_new_opportunity` as `llm_judge` (not `agent_loop`) based on
the assumption that the scoring skill contains a bounded rubric (hot/warm/cold/disqualified).
The signature stub and prompt exist, but the rubric section is a `[PLACEHOLDER]`.

**Action required:** Before deploying this pipeline, inline the full scoring rubric from
`~/.claude/skills/deal-scoring/SKILL.md` into `signatures/score_new_opportunity.py`
(the `SCORE_OPP_PROMPT` constant) and into `pipeline.yaml`
(`nodes[score_new_opportunity].signature_spec.prompt`). Then run `rote compile --update`
on just this node to re-derive it with the rubric in scope.

If the scoring skill turns out to be an open-ended research step (not a rubric), reclassify
`score_new_opportunity` as `agent_loop` with `tools: [slack, gmail]` and set a
`termination.max_iterations: 1`.

---

### Q-2 (MEDIUM): Fuzzy match threshold of 80 may need tuning

**What I decided:** `FUZZY_MATCH_THRESHOLD = 80` in `extracted/matching.py`. This is a
reasonable industry default for partial string matching (rapidfuzz `partial_ratio`).

**Risk:** With short account names like "Nike" vs "Nike US", partial ratio can be too generous.
With names like "Blue Horizon Outdoor Apparel, LLC" vs "Blue Horizon Outdoor", it may be too
strict.

**Recommended next step:** Run the first 10 real morning runs and log every fuzzy match
decision (matched_deal, thread_account_name, score). Tune the threshold based on false
positives / negatives observed.

---

### Q-3 (MEDIUM): `extract_thread_data` binding of `known_account_names` is unbound

**What I decided:** In the `extract_thread_data` node's `inputs:`, `known_account_names` is
noted as unbound — the grammar can only reference whole node outputs or top-level fields, not
derived lists like "account names extracted from a list of DealRecord objects."

**What the implementation must do:** The adapter or runtime harness that invokes
`extract_thread_data` needs to pass `apply_inclusion_filter.output.included` as the deal list
and derive account_name strings from it. This is a small amount of glue code in the
workflow orchestration layer.

---

### Q-4 (LOW): No HITL gate despite "present the output" in Step 6

**What I decided:** Step 6 says "Present the HTML file for download." I classified this as a
terminal `pure_function` (`generate_summary`) + `external_call` (`save_dashboard`), not a
`hitl_gate`. The skill doesn't say "wait for approval" — it just produces output.

**Risk:** If the intent was for the account owner to review the dashboard and approve next
actions before the workflow triggers anything downstream, a `hitl_gate` after `generate_summary`
would be appropriate. But since this pipeline has no downstream action (no email send, no CRM
write), the gate would suspend indefinitely with no continuation.

**Recommendation:** If a downstream action is added (e.g., "send the Slack summary to
#deal-updates"), add a `hitl_gate` node before that action with
`signal: dashboard_reviewed`.

---

### Q-5 (LOW): Schedule defaulted to weekdays 8 AM — verify timezone

**What I decided:** `config.schedule: "0 8 * * 1-5"` — weekdays at 8 AM.

**Verify:** The cron runs in the workflow engine's timezone. Confirm whether this is UTC or
the operator's local time (likely US/Eastern or US/Pacific based on the NA/EU deal geography).
Adjust to `"0 13 * * 1-5"` for 8 AM Eastern in UTC, for example.

---

## Suggested Next Steps

1. **Inline the deal-scoring rubric (blocking).** See Q-1. Without it, the `score_new_opportunity`
   node produces unreliable scores. Everything else in the pipeline is production-ready.

2. **Wire the MCP clients.** Both `extracted/slack.py` and `extracted/gmail.py` raise
   `NotImplementedError`. Implement with the Slack and Gmail MCP tools already available in the
   source skill environment. Both are simple tool calls — each implementation should be under
   20 lines.

3. **Dogfood first on `classify_thread_status`.** This node processes threads that already exist
   in Gmail. Run the classifier on 10-20 real threads, compare to what a human would classify,
   and use the disagreements to extend `evals/classify_thread_status.jsonl`. This is the highest-
   ROI eval to run first because classification accuracy directly determines what shows up in the
   dashboard's "needs attention" row.

4. **Run `rote eval` to get the before/after scorecard.** The `eval.yaml` sidecar is written.
   Run `rote eval examples/deal-monitor/` to compute the estimated cost reduction. Expected
   outcome: ~60–70% fewer LLM calls per run, since 4 out of 12 steps need the LLM (and many
   of those were previously done ad-hoc inside the agent loop).

5. **Add the warehouse domain list.** `extracted/gmail.py` has a `TODO` for real warehouse
   contact domains (search 4). This should be populated with the actual warehouse 3PL email
   domains used in outreach so search 4 catches threads from warehouse-initiated replies.

6. **Tune the fuzzy match threshold (Q-2)** after the first 10 real runs.

7. **Consider a `hitl_gate` if downstream actions are added** — see Q-4.
