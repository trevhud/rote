# Graduation Report — deal-monitor

## Summary Metrics

| Metric | Value |
|---|---|
| Total nodes | 10 |
| `pure_function` | 4 (`filter_and_extract_opps`, `match_opps_to_threads`, `render_dashboard`, `generate_summary`) |
| `external_call` | 4 (`fetch_slack_messages`, `search_gmail_fixed`, `search_gmail_by_account`, `write_dashboard`) |
| `llm_judge` | 2 (`classify_warehouse_thread`, `score_new_opportunity`) |
| `agent_loop` | 0 |
| `hitl_gate` | 0 |
| % codifiable | 80% (8 of 10 non-hitl nodes) |
| HITL gates | None — skill has no human approval step |
| Estimated agent turns before graduation | 20–90 per run |
| LLM calls in graduated pipeline | 2 fan-out judges (N threads + M unmatched opps per run) |

**The biggest single win:** the source skill serialized Slack intake and Gmail searches even though they share zero data. The graduated pipeline puts both in the same wave — wall-clock time for those two steps drops by roughly half.

**Token cost reduction:** the four `pure_function` nodes (filtering, matching, rendering, summary) consumed most of the agent's context on every run. They are now deterministic Python. The `render_dashboard` HTML generation step was particularly wasteful in the agent — the model was re-deriving a fixed template from prose. It's now a 0-token pure function.

---

## Crystallization Log

### 1. Slack channel ID and message limit → constants
**Source:** SKILL.md Step 1 — "50 most recent messages from #deal-intake (channel ID: `C0EXAMPLE000`)"

**Before:** Agent reads the channel ID and limit from prose on every run.

**After:**
```python
SLACK_CHANNEL_ID: str = "C0EXAMPLE000"
SLACK_MESSAGE_LIMIT: int = 50
```
in `extracted/slack.py`. The channel ID can never silently drift.

---

### 2. NA SKU threshold → pre-filter + constant
**Source:** SKILL.md Step 1 — "Only if SKU count < 2,000. Unknown SKU = include, flag as risk. Confirmed ≥ 2,000 = exclude."

**Before:** Agent applies this rule via prose reading on every message.

**After:**
```python
NA_SKU_THRESHOLD: int = 2_000

# In filter_and_extract_opps:
elif sku_count is not None and sku_count >= NA_SKU_THRESHOLD:
    should_include = False
```
`mandatory: true` on `filter_and_extract_opps` makes this impossible to skip.

---

### 3. Always-include rep names → constant
**Source:** SKILL.md Step 1 — "submitted by Alex Rivers or Sam Patel (always UK)"

**Before:** Agent tries to remember these names from context.

**After:**
```python
EU_UK_ALWAYS_INCLUDE_REPS: list[str] = ["Alex Rivers", "Sam Patel"]
```
A new rep added to this list requires a code change, not a prompt edit — reviewable in git.

---

### 4. Gmail subject filters + lookback window → constants
**Source:** SKILL.md Step 2 — `"Quote Request"`, `"New Business"`, `"RFP"`, `"last 60 days"`

**Before:** Agent constructs 5 search queries from prose each run.

**After:**
```python
GMAIL_SUBJECT_FILTERS: list[str] = ["Quote Request", "New Business", "RFP"]
GMAIL_LOOKBACK_DAYS: int = 60
```
Queries are built deterministically in `extracted/gmail.py`.

---

### 5. Pipeline step names → enum
**Source:** SKILL.md Step 3 — "1=RFP Sent, 2=Response Needed, 3=Follow-Up Needed, 4=Pricing Returned, 5=Pricing Sent, 6=Declined"

**Before:** Agent re-reads this mapping each time it classifies a thread.

**After:**
```python
class PipelineStep(int, Enum):
    RFP_SENT        = 1
    RESPONSE_NEEDED = 2
    FOLLOW_UP_NEEDED = 3
    PRICING_RETURNED = 4
    PRICING_SENT    = 5
    DECLINED        = 6
```
The LLM judge returns a `PipelineStep` value; everything downstream uses the enum.

---

### 6. `needs_attention` → deterministic post-derivation (NOT in LLM judge)
**Source:** SKILL.md Step 5 — "color-coded needs-attention rows"

**Before:** Agent would naturally compute this inline, but it's a hard rule.

**After:**
```python
NEEDS_ATTENTION_STEPS: set[int] = {2, 3}  # Response Needed, Follow-Up Needed
```
Computed in `render_dashboard` without any LLM call. The judge produces `pipeline_step`; the renderer derives attention status. Zero chance of the LLM "forgetting" which steps need attention.

---

### 7. HTML dashboard → fixed template
**Source:** SKILL.md Step 5 — "self-contained HTML file with three tabs: New Opportunities | Quoting | Closed. Use a clean, self-contained layout — no external assets"

**Before:** Agent generates the entire HTML from scratch on every run (high token cost, variable structure).

**After:** `extracted/dashboard.py:render_dashboard` — fixed template with f-string interpolation. 0 LLM tokens for rendering.

---

### 8. Text summary format → fixed template
**Source:** SKILL.md Step 6 — "new opps count, active quoting count (with how many need attention), closed count"

**Before:** Agent writes the summary in natural language, format varies.

**After:**
```python
def generate_summary(dashboard: DashboardData) -> str:
    return f"New opps: {dashboard.new_opp_count} | Quoting: {dashboard.quoting_count}{attention_note} | Closed: {dashboard.closed_count}"
```

---

### 9. Parallel entry nodes (Pattern 8)
**Source:** SKILL.md — Step 1 and Step 2 are listed sequentially in prose but share no data.

**Before:** Agent executes Slack pull, waits for it to finish, then begins Gmail searches.

**After:** `entry_nodes: [fetch_slack_messages, search_gmail_fixed]` — both start in the same wave. The sequential prose ordering was an artifact of single-threaded agent execution.

---

## Open Questions

### 1. Deal-scoring rubric is an external file dependency
**What the skill says:** "Read the deal-scoring skill at `~/.claude/skills/deal-scoring/SKILL.md` and apply its scoring rules."

**What I did:** The `score_new_opportunity` signature accepts `deal_scoring_rubric: str` as an input field, injected at runtime from the path in `pipeline.input.deal_scoring_rubric_path`. This defers rubric loading to the adapter/driver layer.

**Why you should verify:** The deal-scoring skill may itself be graduatable — if it has a structured rubric with discrete tiers, the scoring could be a `pure_function` pre-filter + a simpler `llm_judge`. I have not seen the deal-scoring SKILL.md; it's an open dependency. Check whether it contains numeric thresholds that belong in Python pre-filters.

**Alternative:** Graduate the deal-scoring skill separately and reference it as a subpipeline. This would let the two skills evolve independently and be individually eval'd.

---

### 2. Warehouse contact domains not in the skill
**What the skill says:** Search 4 is "any thread where the sender or recipient is a known warehouse contact domain."

**What I did:** The `search_gmail_fixed` function documents a `WAREHOUSE_CONTACT_DOMAINS` env var / rote config placeholder. The list of domains is not in the skill text.

**Why you should verify:** These domains need to come from somewhere. Options: (a) add `warehouse_contact_domains` to the pipeline input (already in `input.optional`), (b) maintain a static list in `extracted/gmail.py`, or (c) fetch from a contacts database. The graduation defaults to (a) — pass domains at runtime.

---

### 3. "Closed" tab definition
**What the skill says:** Three tabs — "New Opportunities | Quoting | Closed." The Closed tab is not explicitly defined.

**What I assumed:** Closed = threads classified as step 6 (Declined). This is the only terminal state in the 6-step pipeline. But "Closed" could also include won deals (fully enrolled) if such a state exists.

**Why you should verify:** Confirm with the account owner whether Closed means "Declined" only, or if there's a "Pricing Accepted / Closed Won" state that should be a step 7 in the pipeline classification.

---

### 4. Gmail search 4 semantics (warehouse contact domains)
**What the skill says:** "any thread where the sender or recipient is a known warehouse contact domain."

**What I did:** The Gmail query pattern `from:WAREHOUSE_DOMAINS OR to:WAREHOUSE_DOMAINS` is a placeholder — actual Gmail query syntax uses `from:domain.com`, not a list operator. The `search_gmail_fixed` stub documents this; the implementer needs to expand it per-domain.

**Why you should verify:** If the warehouse domain list is large (50+ domains), individual queries per domain will be slow and may hit rate limits. A better approach: maintain a Gmail label for warehouse threads and search by label instead.

---

## Suggested Next Steps

1. **Dogfood with real Slack/Gmail data.** The hardest failure mode to discover analytically is thread-to-opp matching: fuzzy match threshold 80 (token_sort_ratio) may produce false positives when account names are short or ambiguous. Run on a week's real data and measure precision/recall on the match step.

2. **Eval `classify_warehouse_thread` first.** It has the most concrete ground truth — the 6-step definition is unambiguous. Run the 6 seed examples in `evals/classify_warehouse_thread.jsonl`, expand to 10–15 real threads, then DSPy-compile to lock in few-shots.

3. **Get the deal-scoring skill and graduate it.** The `score_new_opportunity` node carries a `deal_scoring_rubric` string that the LLM reads inline — if the scoring rubric has explicit numeric thresholds or tier criteria, those belong in a pre-filter. Graduating the deal-scoring skill in isolation will reduce the prompt size for every scoring call.

4. **Consider haiku for both judges.** The two `llm_judge` nodes are already defaulted to `claude-haiku-4-5-20251001` — fast and cheap for bounded classification. If evals show accuracy issues, upgrade to Sonnet on the problem judge only rather than both.

5. **Re-graduate `search_gmail_fixed` if warehouse domain list changes.** The search 4 placeholder is the weakest part of the graduation — once the actual domain list is known, this node should be updated to expand one Gmail query per domain and potentially be split into a `fan_out: true` node.

---

## Node Summary

| Node | Kind | Mandatory | MCP Binding | Parallel Wave |
|---|---|---|---|---|
| `fetch_slack_messages` | `external_call` | ✅ | slack / slack_read_channel | Wave 1 (entry) |
| `search_gmail_fixed` | `external_call` | ✅ | gmail / gmail_search_threads | Wave 1 (entry) |
| `filter_and_extract_opps` | `pure_function` | ✅ | — | Wave 2 |
| `search_gmail_by_account` | `external_call` | ✅ | gmail / gmail_search_threads | Wave 3 |
| `match_opps_to_threads` | `pure_function` | ✅ | — | Wave 4 |
| `classify_warehouse_thread` | `llm_judge` | ✅ | — | Wave 5 (parallel) |
| `score_new_opportunity` | `llm_judge` | ✅ | — | Wave 5 (parallel) |
| `render_dashboard` | `pure_function` | ✅ | — | Wave 6 |
| `write_dashboard` | `external_call` | ✅ | — | Wave 7 (parallel) |
| `generate_summary` | `pure_function` | ✅ | — | Wave 7 (parallel) |
