# BDR Outreach — Compilation Report

**Skill:** `examples/bdr-outreach/skill/SKILL.md`
**Compiled:** 2026-04-07
**Compiler:** rote-compile v0.1

---

## Summary Metrics

| Metric | Value |
|---|---|
| Total nodes | 22 |
| `pure_function` nodes | 5 |
| `external_call` nodes | 10 |
| `llm_judge` nodes | 2 |
| `agent_loop` nodes | 2 |
| `hitl_gate` nodes | 3 |
| Codifiable (PF + EC) | 15 / 19 non-gate nodes = **79%** |
| HITL gates | 3 (contact_review_gate, conf_exclusion_review_gate, manual_enrollment_handoff) |
| MANDATORY checks codified | 3 (DNC, recently emailed, active sequence) |
| Extracted Python modules | 6 |
| LLM judge signatures | 2 |
| Eval seed examples | 8 (vet_contact) + 4 (personalize_email) |

**Estimated token reduction per campaign run:** 40–60%

The primary savings come from:
1. Moving the `is_pharma()` classifier, batching loops, and report templates out of the agent context
2. Pre-filtering ~20–30% of contacts in `vet_contact` before the LLM is called
3. Replacing all three MANDATORY exclusion checks with direct API calls (no LLM reasoning)
4. Caching taxonomy IDs for 30 days (saves 4 ZoomInfo API calls + LLM overhead on every run after first)

The 79% codifiability score is above the 60–70% target, which is expected: the BDR skill has unusually explicit procedural descriptions (literal Python code, exact batch sizes, verbatim report templates). The remaining 21% (`target_research`, `lead_generation_loop`, `vet_contact`, `personalize_email`) are genuinely fuzzy — they depend on reading prose, exploring unfamiliar sources, and applying judgment that varies by contact.

---

## HITL Gates

| Gate | Signal | Timeout | What It Blocks |
|---|---|---|---|
| `contact_review_gate` | `contact_review_approved` | 7d | CRM upload; BDR may add/remove/reprioritize contacts |
| `conf_exclusion_review_gate` | `exclusion_review_confirmed` | 2d | Conference enrichment; BDR confirms non-pharma exclusions |
| `manual_enrollment_handoff` | `bdr_enrollment_complete` | 14d | Pipeline exit; BDR manually enrolls in HubSpot UI |

**Why manual_enrollment_handoff is a gate and not a terminal node:** the skill explicitly states "Sequence enrollment must be done manually in the HubSpot UI — there is no safe API for this (API enrollment sends emails immediately with no verification step)." The gate records that enrollment happened and closes the workflow audit trail.

---

## Phase 2: Node Classification Table

| Node | Kind | Justification |
|---|---|---|
| `taxonomy_lookup` | `external_call` | Fixed ZoomInfo API calls for stable IDs; 4 lookups with known params; cacheable 30d |
| `target_research` | `agent_loop` | Genuinely exploratory: different queries per company/indication, variable sources, termination based on sufficient intel coverage |
| `lead_generation_loop` | `agent_loop` | Variable backfill searches, quota-based termination, search terms vary run-over-run |
| `enrich_contact_batch` | `external_call` | Fixed field set, fixed batch size of 10, deterministic ZoomInfo API call |
| `vet_contact` | `llm_judge` | Reading prose employment history against fuzzy rubric; output bounded (decision, tier, reason, evidence) |
| `build_contact_table` | `pure_function` | Fixed markdown table + narrative template; all inputs are already structured counts |
| `contact_review_gate` | `hitl_gate` | Skill explicitly says "present to user... the user may add, remove, or reprioritize" |
| `conf_load_file` | `pure_function` | File parsing is deterministic; fixed column mapping |
| `conf_pharma_filter` | `pure_function` | Literal Python `is_pharma()` function in the skill; keyword matching is deterministic |
| `conf_exclusion_review_gate` | `hitl_gate` | Skill says "always show the user the excluded list and ask if any companies should be moved back to pharma" |
| `conf_enrich_batch` | `external_call` | Same as `enrich_contact_batch`; smaller output field set (no employment history) |
| `conf_build_xlsx` | `pure_function` | Literal openpyxl code in the skill; fixed column widths, colors, formatting |
| `hubspot_upsert` | `external_call` | Fixed API, batch_size=100 per HubSpot limit, deterministic upsert-by-email semantics |
| `hubspot_create_list` | `external_call` | Fixed API, naming convention documented in skill |
| `hubspot_add_to_list` | `external_call` | Fixed API, batch_size=250 per HubSpot limit |
| `exclusion_check_dnc` | `external_call` (mandatory) | Fixed sequence: search DNC list ID → check memberships per contact; no LLM reasoning |
| `exclusion_check_recent` | `external_call` (mandatory) | Fixed API: daysBack=30 hardcoded; `wasEmailedInPeriod` boolean result |
| `exclusion_check_sequence` | `external_call` (mandatory) | Fixed API: `isEnrolled` boolean result per contact |
| `personalize_email` | `llm_judge` | Drafting personalized copy requires reading contact history + intel brief; output bounded (subject, body_html, hook) |
| `hubspot_save_template` | `external_call` | Fixed internal API with known parameters; cookie-based auth is operational concern |
| `pre_enrollment_report` | `pure_function` | Verbatim template from hubspot-operations.md lines 86–103; fills in counts from structured inputs |
| `manual_enrollment_handoff` | `hitl_gate` | Enrollment must be done manually; workflow waits for BDR signal |

---

## Phase 3: Crystallization Log

### Extraction 1 — `is_pharma()` function

**Source:** `conference-enrichment.md:52–84`

**Before (prose-to-code):**
```
Write a Python script that classifies each contact's company as pharma/biotech or not.
Use keyword matching on the company name.
Pharma/Biotech keywords to INCLUDE (case-insensitive): pharma, biopharm, biotech, ...
Non-Pharma keywords to EXCLUDE: consulting, consultancy, ...
def is_pharma(company_name):
    ...
```
The LLM read this function definition and mentally executed it on each contact — on every run.

**After:** `extracted/conference_filter.py:is_pharma()` — lifted verbatim with `INCLUDE_KEYWORDS` and `EXCLUDE_KEYWORDS` as module-level constants.

**Confidence:** HIGH — exact extraction from code fence.

---

### Extraction 2 — `extract_linkedin()` function

**Source:** `conference-enrichment.md:109–115`

**Before:** LLM parsed the `externalUrls` array and found LinkedIn URLs each time.

**After:** `extracted/conference_filter.py:extract_linkedin()` — verbatim extraction.

**Confidence:** HIGH.

---

### Extraction 3 — ZoomInfo enrich batching loop

**Source:** `conference-enrichment.md:126–135`, `lead-generation.md:92–101`

**Before:**
```python
BATCH_SIZE = 10
for i in range(0, len(pharma_contacts), BATCH_SIZE):
    batch = pharma_contacts[i:i+BATCH_SIZE]
    ...
```
LLM re-derived this loop and the batch size on every conference run.

**After:** `extracted/zoominfo.py:enrich_contacts_batch()` with `ENRICH_BATCH_SIZE = 10`.

**Constants lifted:** `ENRICH_BATCH_SIZE = 10`, `ENRICH_OUTPUT_FIELDS = [...]`

**Confidence:** HIGH — batch size appears in skill limits table AND lead-generation.md AND conference-enrichment.md (three independent sources, all say 10).

---

### Extraction 4 — openpyxl XLSX builder

**Source:** `conference-enrichment.md:143–174`

**Before:** LLM generated openpyxl formatting code (colors, column widths, freeze panes) from the prose spec.

**After:** `extracted/conference_xlsx.py:build_conference_xlsx()` with constants:
```python
HEADER_COLOR = "1F4E79"
ALT_ROW_COLOR = "EBF3FB"
COLUMNS = [("Company", "Company", 30), ("First Name", ..., 18), ...]
FILENAME_TEMPLATE = "{year}_{conference_name}_Pharma_Contacts.xlsx"
```

**Confidence:** HIGH — exact hex values, column widths, and filename format are in the skill prose.

---

### Extraction 5 — HubSpot batch sizes (MANDATORY limits)

**Source:** `SKILL.md:149–157` (limits table), `hubspot-operations.md:7,21`

**Constants lifted:**
- `UPSERT_BATCH_SIZE = 100` (HubSpot batch upsert limit)
- `LIST_ADD_BATCH_SIZE = 250` (HubSpot add-to-list limit)
- `ENRICH_BATCH_SIZE = 10` (ZoomInfo enrich limit)

These limits are API constraints that cannot drift without breaking the integration. Moving them to code makes violations a runtime ValueError, not a silent mismatch.

**After:** Module-level constants in `extracted/hubspot.py` and `extracted/zoominfo.py`.

**Confidence:** HIGH — appear in multiple places in the skill; API limits are well-documented.

---

### Extraction 6 — MANDATORY exclusion checks

**Source:** `hubspot-operations.md:48` (header "Exclusion Checks (MANDATORY)"), `SKILL.md:77` ("Phase 5: Exclusion Checks (MANDATORY)"), `hubspot-operations.md:49` ("always run these checks")

**Before:** Three prose-described checks that the LLM was expected to run in order, with no enforcement mechanism. Prompt drift or long agent trajectories could silently skip them.

**After:** Three `mandatory: true` external_call nodes in the pipeline:
- `exclusion_check_dnc` — `extracted/exclusion_checks.py:check_do_not_contact`
- `exclusion_check_recent` — `extracted/exclusion_checks.py:check_recently_emailed` (constant: `RECENT_EMAIL_DAYS = 30`)
- `exclusion_check_sequence` — `extracted/exclusion_checks.py:check_active_sequence`

The adapter emits these as unconditional activities that cannot be made conditional or skipped.

**Constants lifted:** `RECENT_EMAIL_DAYS = 30`, `DNC_LIST_QUERY = "BDR do not contact"`

**Confidence:** HIGH — "MANDATORY" in ALL CAPS, "always run these checks", `RECENT_EMAIL_DAYS = 30` hardcoded in prose pseudocode.

---

### Extraction 7 — Pre-enrollment report template

**Source:** `hubspot-operations.md:88–103`

**Before:** LLM generated this exact markdown on every run by reading the template in the reference file.

```text
Campaign: Denver Conference 2026
Total contacts enriched: 47
...
Ready to enroll: 35
-> BDR should enroll these contacts manually in HubSpot UI
```

**After:** `extracted/report.py:generate_pre_enrollment_report()` — fixed template with typed parameters.

**Confidence:** HIGH — template is verbatim in the skill, no judgment calls in the output format.

---

### Extraction 8 — Contact table template

**Source:** `lead-generation.md:124–133`

**Before:** LLM generated the ranked contact table markdown with headers, rows, narrative, and discard summary from scratch.

**After:** `extracted/report.py:build_contact_table()` — fixed table structure, tier sorting, narrative format from typed inputs.

**Confidence:** HIGH — table headers, narrative structure, and discard summary format are verbatim in the skill.

---

### Extraction 9 — Accuracy threshold pre-filter

**Source:** `quality-and-vetting.md:42`, `lead-generation.md:108`

**Before:** The rubric said "Flag contacts below 85" — the LLM was expected to enforce this as part of the vetting judgment.

**After:** Pre-filter in `signatures/vet_contact.py:VetContact.forward()`:
```python
MIN_ACCURACY_SCORE: int = 85
if inputs.contact.contact_accuracy_score < MIN_ACCURACY_SCORE:
    return VetContactOutput(decision=VetDecision.DISCARD, discard_reason=DiscardReason.LOW_ACCURACY, ...)
```
Contacts below 85 never reach the LLM — saves tokens and makes the rule undriftable.

**Confidence:** HIGH — 85 appears in both files independently.

---

## Phase 4: LLM Judge Signatures

### `vet_contact` — `signatures/vet_contact.py:VetContact`

**Rubric source:** `quality-and-vetting.md`

**Decision space:**
- `VetDecision`: `keep` | `discard`
- `ContactTier`: `ideal` | `strong` | `good` (only when keep)
- `DiscardReason`: 11 enum members (one per named red-flag category + low_accuracy + no_valid_email)
- `relevance_evidence`: free string (1–2 sentences)

**Pre-filters:** accuracy < 85 → `low_accuracy` discard; no valid email → `no_valid_email` discard. Both short-circuit before the LLM is called. Expected to handle ~20–30% of contacts without any model call.

**Seed eval set:** `evals/vet_contact.jsonl` — 8 examples covering all major decision paths: msl_role, biomarker_discovery, low_accuracy (pre-filter), no_valid_email (pre-filter), indication_mismatch, sales_commercial, ideal tier, strong tier.

---

### `personalize_email` — `signatures/personalize_email.py:PersonalizeEmail`

**Rubric source:** `email-templates.md`

**Decision space:**
- `subject`: string (one line)
- `body_html`: HubSpot HTML, < 200 words
- `campaign_type_used`: `drug-specific` | `condition-specific` | `general-capabilities`
- `personalization_hook`: string (1 sentence, the lead observation used)

**Note:** The output is less tightly bounded than `vet_contact` — `body_html` is essentially free text within structural constraints. This is appropriate here because the personalization requires genuine creativity. The `personalization_hook` field provides the one structured piece for tracing and eval.

**Seed eval set:** `evals/personalize_email.jsonl` — 4 examples: drug-specific ideal, condition-specific strong, general-capabilities good, and a word-count constraint check.

---

## Phase 6: Adapter Validation Checklist

| Check | Status | Note |
|---|---|---|
| Every `pure_function` / `external_call` has `impl:` | ⚠️ PARTIAL | 3 functions referenced but not yet implemented (see Open Questions) |
| Every `llm_judge` has `signature:` | ✅ | `vet_contact.py`, `personalize_email.py` |
| Every `agent_loop` has `tools:` | ✅ | `target_research`, `lead_generation_loop` |
| Every `hitl_gate` has `signal:` | ✅ | All 3 gates have signals |
| All edge node IDs exist in `nodes:` | ✅ | Verified manually |
| All `loop_body` IDs exist in `nodes:` | ✅ | `enrich_contact_batch`, `vet_contact` |
| All `entry_nodes` exist in `nodes:` | ✅ | `taxonomy_lookup`, `target_research`, `conf_load_file` |
| All `exit_nodes` exist in `nodes:` | ✅ | `manual_enrollment_handoff` |
| No YAML key named `on:` | ✅ | Used `retry_on:` and `on_signal:` throughout |
| `mandatory: true` not on `agent_loop` | ✅ | Only on `external_call` nodes |

---

## Open Questions

### OQ-1: Three stub functions not yet implemented

Three functions are referenced in `impl:` fields but not yet in the extracted modules:

1. **`extracted/zoominfo.py:lookup_taxonomy_ids`** — The taxonomy lookup step runs 4 parallel ZoomInfo calls. The function should parallelize them (e.g., `asyncio.gather`) and return a `TaxonomyIds` dataclass. The skill's lead-generation.md lines 19–25 describe the exact 4 calls.

2. **`extracted/hubspot.py:save_sales_template`** — Should call `hubspot_search_sales_templates` to check if a template with the given name exists, then either `hubspot_create_sales_template` or `hubspot_update_sales_template`. Cookie-based auth is a pre-condition.

3. **`extracted/conference_filter.py:load_attendee_file`** — Should detect file type (CSV, XLSX, Google Sheet URL) and parse into a normalized list of dicts with standardized keys. The skill mentions CSV, XLSX, and Google Sheet but doesn't give code for this step.

**Decision made:** Marked as stubs and flagged for first-sprint implementation. The IR is otherwise complete.

### OQ-2: Conditional routing between standard and conference paths

The IR DAG has two separate entry paths (standard: `[taxonomy_lookup, target_research]`; conference: `[conf_load_file]`) that both converge at `hubspot_upsert`. The IR schema has no explicit conditional-edge semantics — the adapter must implement routing based on `campaign_type` in the pipeline input.

**Decision made:** Documented as an adapter responsibility. The `campaign_type: "conference"` value should route to `conf_load_file` and skip the standard path nodes. The adapter should treat the two paths as mutually exclusive branches that converge at `hubspot_upsert`.

**Alternative:** Split into two separate `pipeline.yaml` files (`bdr-standard.yaml` and `bdr-conference.yaml`) sharing nodes from Phase 4 onward via a shared library. This would make the IR fully DAG-shaped without routing logic. Recommend evaluating after the first adapter is implemented.

### OQ-3: `personalize_email` classified as `llm_judge` — verify this is right

The `personalize_email` node has a partially bounded output (`body_html` is free text within structural constraints). It was kept as `llm_judge` rather than `agent_loop` because:
- The output schema is defined (subject, body_html, campaign_type_used, personalization_hook)
- The output is generated from a single bounded input (one contact's data + brief + intel)
- The base template structure is fixed and injected as system context

If the first production run reveals that the personalization judgment requires iterative tool calls (e.g., fetching additional company context), reclassify as `agent_loop`.

### OQ-4: `acme_experience` input to `personalize_email` — sourcing not implemented

The email templates reference says "Acme's experience in specific indications... must come from validated internal sources only." The `personalize_email` signature has an `acme_experience: str` input field, but the pipeline has no node that populates it — it's expected to come from the `target_research` agent loop's internal source queries.

**Decision made:** The `target_research` node should be extended to also query the Acme publications/research database and include validated TA experience in its `IntelBrief` output. The `IntelBrief.acme_experience` field should be populated there. This prevents the email personalization LLM from fabricating capabilities.

### OQ-5: HubSpot cookie auth for `hubspot_save_template`

The skill notes that internal API tools require browser session cookies that expire periodically. The `hubspot_save_template` node's `retry_on: [network]` does NOT handle auth expiration. The adapter should implement a cookie-refresh hook (the skill mentions a `rotate-hubspot-cookie` script) that runs before the retry. Otherwise the node will fail silently on cookie expiration.

### OQ-6: `conf_build_xlsx` not in `exit_nodes` — adapter must not wait on it

`conf_build_xlsx` is a side-artifact terminal node with no outbound edges. The pipeline continues through `hubspot_upsert` in parallel. The adapter must not wait for `conf_build_xlsx` to complete before proceeding to `hubspot_upsert` — they run as independent branches from `conf_enrich_batch`.

---

## Suggested Next Steps

### 1. Implement the three stub functions (OQ-1) — first sprint

`lookup_taxonomy_ids`, `save_sales_template`, and `load_attendee_file` are the only gaps preventing the adapter from generating runnable code. Each is < 50 lines of straightforward Python.

### 2. Run the vet_contact eval set against a real model — highest leverage

The 8 seed examples in `evals/vet_contact.jsonl` cover all major decision paths. Running them against a live model will reveal immediately whether the pre-filter thresholds are right and whether the LLM correctly applies the indication-mismatch and MSL-role rules (the two most commonly confused categories in BDR history).

After 10–20 real campaign runs, use DSPy's `BootstrapFewShot` or BAML `test` to optimize the signature prompt against the accumulated eval set.

### 3. Dogfood the conference path first — lowest blast radius

The conference path (`conf_load_file` → `conf_pharma_filter` → ... → `conf_build_xlsx`) is almost entirely `pure_function` and `external_call` nodes. It has no agent loop. Running a real conference list through the compiled pipeline (without going all the way to CRM upload) will validate the extracted Python modules and XLSX builder without touching live HubSpot data.

### 4. Instrument `vet_contact` discard reasons in production

Once live, log the `discard_reason` enum values for every discarded contact. After 500+ contacts, build a histogram. This tells you:
- Which discard categories are most common (→ optimize pre-filter or add more examples)
- Whether `indication_mismatch` is catching false positives (→ the franchise-alignment check is subtle)
- Whether `other` is too high (→ the rubric may have uncovered edge cases that need new enum values)

### 5. Recompile `lead_generation_loop` after 3 campaign runs

The initial 3 searches (ideal persona, manufacturer team, TA broad net) are fixed procedures that could be modeled as 3 parallel `external_call` nodes feeding the enrichment batch. The only part that requires `agent_loop` is the backfill logic. After 3 real runs, you'll know whether the initial searches consistently exhaust the first round without backfill — if yes, split the node.

### 6. Evaluate splitting into two pipelines (OQ-2)

If the conditional routing between standard and conference paths proves awkward in the first adapter, split `bdr-outreach.yaml` into `bdr-lead-gen.yaml` (Phases 1–7 standard) and `bdr-conference.yaml` (Phases 2-alt + 4–7). Both share the Phase 4–7 nodes. This makes both pipelines pure DAGs with no routing logic.
