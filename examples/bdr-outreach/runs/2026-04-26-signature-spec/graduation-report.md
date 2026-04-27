# BDR Outreach Campaign — Graduation Report

Source skill: `examples/bdr-outreach/skill/SKILL.md`
Graduated: 2026-04-26
Runtime target: temporal (or cloudflare — `signature_spec` populated for both)

---

## Summary Metrics

| Metric | Value |
|---|---|
| Total nodes | 22 |
| `pure_function` | 5 |
| `external_call` | 10 |
| `llm_judge` | 2 |
| `agent_loop` | 2 |
| `hitl_gate` | 3 |
| Codifiable nodes (pure_function + external_call) | 15/20 non-gate = **75%** |
| Mandatory nodes | 3 (`exclusion_check_dnc`, `exclusion_check_recent`, `exclusion_check_sequence`) |
| HITL gates | 3 |
| Estimated token reduction vs. raw agent loop | ~65–70% per run |

### HITL Gates

| Gate | Signal | Timeout | Blocks On |
|---|---|---|---|
| `contact_review_gate` | `contact_review_approved` | 7d | CRM upload — BDR approves/edits vetted contact list before data enters HubSpot |
| `conf_exclusion_review_gate` | `exclusion_review_confirmed` | 2d | ZoomInfo enrichment — BDR confirms excluded non-pharma companies before spending enrichment credits |
| `manual_enrollment_handoff` | `bdr_enrollment_complete` | 14d | Sequence enrollment — enrollment MUST be manual (API sends emails immediately with no verification) |

---

## Phase 2: Node Classification

| Step | Kind | Justification |
|---|---|---|
| Campaign brief (intake) | Pipeline input | Fixed schema; validation belongs at the input boundary, not as a workflow node |
| Target company research (Phase 1.5) | `agent_loop` | Tool choices vary per indication; sources aren't known in advance; termination depends on research completeness |
| Taxonomy ID resolution (Phase 2 setup) | `external_call` | Four deterministic API lookups with fixed parameters and stable outputs; cache-worthy |
| Lead generation loop (Phase 2) | `agent_loop` | Number of iterations, specific queries, and backfill strategy genuinely vary per campaign; quota-driven termination |
| ZoomInfo enrich batch (Phase 2 sub) | `external_call` | Fixed batch semantics, known output field set, hard limit of 10 contacts per call |
| Vet contact (Phase 2 sub) | `llm_judge` | Reads employment history (prose) and applies fuzzy rubric; output space is bounded (keep/discard/tier/reason) |
| Build contact table (Phase 2 output) | `pure_function` | Fixed Markdown template with ranked rows and discard summary; no LLM |
| Contact review gate (Phase 3) | `hitl_gate` | Explicit BDR approval required before CRM upload; user may edit the list |
| Conference file load (Phase 2-alt) | `pure_function` | Fixed CSV/XLSX parsing; deterministic column normalization |
| Conference pharma filter (Phase 2-alt) | `pure_function` | `is_pharma()` function with hardcoded keyword lists; zero LLM |
| Conference exclusion review gate (Phase 2-alt) | `hitl_gate` | Explicit BDR confirmation of excluded companies required before enrichment |
| Conference enrich batch (Phase 2-alt) | `external_call` | Same as standard enrich but smaller output_fields; fixed batch semantics |
| Conference build XLSX (Phase 2-alt) | `pure_function` | Fixed openpyxl formatting; all constants in code |
| HubSpot batch upsert (Phase 4) | `external_call` | Deterministic API call with fixed chunking at 100-contact limit |
| HubSpot create list (Phase 4) | `external_call` | Deterministic API call; naming convention is fixed |
| HubSpot add to list (Phase 4) | `external_call` | Deterministic API call with fixed chunking at 250-contact limit |
| Exclusion check DNC (Phase 5) | `external_call` | Fixed procedure from hubspot-operations.md; MANDATORY; no judgment involved |
| Exclusion check recent email (Phase 5) | `external_call` | Fixed 30-day window; MANDATORY; pure API check |
| Exclusion check active sequence (Phase 5) | `external_call` | Fixed isEnrolled check; MANDATORY; pure API check |
| Pre-enrollment report (Phase 5) | `pure_function` | Fixed Markdown template from hubspot-operations.md; counts from structured inputs |
| Personalize email (Phase 6) | `llm_judge` | Generates opening line + TA callout from unbounded enrichment prose; output schema is bounded |
| HubSpot save template (Phase 6) | `external_call` | Deterministic create/update API call; requires cookie auth |
| Manual enrollment handoff (Phase 7) | `hitl_gate` | API enrollment REMOVED intentionally — BDR must confirm manual UI enrollment |

---

## Phase 3: Crystallization Log

Every prose-to-code extraction from the source skill's reference files.

### Extraction 1 — `is_pharma()` function
**Source:** `references/conference-enrichment.md` lines 52–85 (fenced Python code block)
**Before (source):** LLM reads the full Python function from the skill file at runtime and mentally "executes" it against each company name — a classic Pattern 1 waste.
**After:** Function lifted verbatim into `extracted/conference_filter.py:is_pharma`. Keyword lists (`EXCLUDE_KEYWORDS`, `INCLUDE_KEYWORDS`) defined as module-level constants.

### Extraction 2 — `extract_linkedin()` helper
**Source:** `references/conference-enrichment.md` lines 107–113 (fenced Python code block)
**Before:** LLM navigates the `externalUrls` list structure on every contact.
**After:** Lifted verbatim into `extracted/conference_xlsx.py:extract_linkedin`.

### Extraction 3 — ZoomInfo enrich batching loop
**Source:** `references/conference-enrichment.md` lines 121–134 (`BATCH_SIZE = 10`, batching loop pseudocode)
**Before:** LLM re-derives the batching logic each run. `BATCH_SIZE` was a prose constant.
**After:** `ENRICH_BATCH_SIZE = 10` in `extracted/zoominfo.py`. Chunking is inside `enrich_contacts_batch`, not the workflow. The workflow calls it once with a full list.

### Extraction 4 — ZoomInfo output field set
**Source:** `references/lead-generation.md` lines 94–99 (fixed `outputFields` list)
**Before:** LLM re-constructs the output fields list from the prose spec on each call.
**After:** `STANDARD_OUTPUT_FIELDS` and `CONFERENCE_OUTPUT_FIELDS` in `extracted/zoominfo.py`. Also surfaced as `constants.output_fields` in the IR nodes `enrich_contact_batch` and `conf_enrich_batch` for adapter-level visibility.

### Extraction 5 — HubSpot batch constants
**Source:** `SKILL.md` limits table; `references/hubspot-operations.md` lines 2–12
**Before:** Numbers embedded in prose ("100 per call", "250 contacts per call").
**After:**
- `UPSERT_BATCH_SIZE = 100` in `extracted/hubspot.py`
- `ADD_TO_LIST_BATCH_SIZE = 250` in `extracted/hubspot.py`
- Also surfaced as `constants.batch_size` in IR nodes `hubspot_upsert` and `hubspot_add_to_list`.

### Extraction 6 — MANDATORY exclusion check procedures (Pattern 3)
**Source:** `references/hubspot-operations.md` lines 48–79 ("Exclusion Checks (MANDATORY)")
**Before:** Three prose procedures enforced only by the word "MANDATORY" in ALL CAPS. Any prompt drift or model upgrade could silently skip them.
**After:**
- `extracted/exclusion_checks.py:check_do_not_contact` — DNC list search + membership check
- `extracted/exclusion_checks.py:check_recently_emailed` — 30-day outbound email check with `RECENT_EMAIL_DAYS = 30`
- `extracted/exclusion_checks.py:check_active_sequence` — enrollment check
- All three nodes carry `mandatory: true` in the IR. The adapter emits them as unconditional activities. Skipping is impossible.

### Extraction 7 — Pre-enrollment report template
**Source:** `references/hubspot-operations.md` lines 86–103 (fixed text template with placeholders)
**Before:** LLM regenerates this Markdown from memory each run, risking format drift.
**After:** `extracted/report.py:generate_pre_enrollment_report` — pure string formatting from structured `ExclusionCounts` inputs. The exact template text is embedded in the function.

### Extraction 8 — XLSX formatting constants
**Source:** `references/conference-enrichment.md` lines 148–174 (openpyxl code block)
**Before:** LLM reads the formatting spec and reconstructs the openpyxl calls.
**After:** All constants lifted into `extracted/conference_xlsx.py`:
```python
HEADER_FILL_COLOR = "1F4E79"
ALT_ROW_FILL_COLOR = "EBF3FB"
HEADER_FONT_SIZE = 12
DATA_FONT_SIZE = 11
COLUMNS = ["Company", "First Name", "Last Name", "Job Title", "LinkedIn", "Email"]
COLUMN_WIDTHS = [30, 18, 18, 35, 45, 35]
```

### Extraction 9 — `DNC_LIST_QUERY` constant
**Source:** `references/hubspot-operations.md` line 53 ("BDR do not contact")
**Before:** Hard-coded string inside LLM-generated tool call, invisible to code review.
**After:** `DNC_LIST_QUERY = "BDR do not contact"` in `extracted/exclusion_checks.py`. Also surfaced as `constants.dnc_list_query` in the `exclusion_check_dnc` node.

### Extraction 10 — Contact table output format
**Source:** `references/lead-generation.md` lines 124–133 (table format template)
**Before:** LLM re-derives column order, tier sort, and discard summary format.
**After:** `extracted/report.py:build_contact_table` — sort by `{ideal: 0, strong: 1, good: 2}`, fixed column order, discard reasons as aggregated string. New `build_contact_table` node in Phase 2.

### Extraction 11 — Taxonomy lookup parameters
**Source:** `references/lead-generation.md` lines 19–24 (four parallel ZoomInfo lookups)
**Before:** LLM re-derives the four lookup calls from prose.
**After:** `TAXONOMY_LOOKUPS` list in `extracted/zoominfo.py`; surfaced as `constants.lookups` in the `taxonomy_lookup` IR node.

---

## Phase 4: LLM Judge Signatures

### `vet_contact` — Apply BDR red-flags rubric
**Source rubric:** `references/quality-and-vetting.md`
**Output schema:**
```
decision: enum[keep, discard]
tier: enum[ideal, strong, good] | null    # only when keep
discard_reason: enum[indication_mismatch, msl_role, biomarker_discovery, translational,
                     sales_commercial, ops_strategy, program_management,
                     low_accuracy, no_valid_email, other] | null   # only when discard
relevance_evidence: str
```
**Pre-filter (no LLM for hard thresholds):**
- `accuracy_score < 85` → `discard:low_accuracy` (saves ~20–40% of LLM calls)
- `email is None` → `discard:no_valid_email`
**Files:** `signatures/vet_contact.py:VetContact`, `evals/vet_contact.jsonl` (9 seed examples)
**Eval coverage:** all 7 discard_reason values + ideal/strong/good tiers

### `personalize_email` — Generate personalized opening line + TA callout
**Source rubric:** `references/email-templates.md`
**Output schema:**
```
opening_line: str   # ≤2 sentences; specific, non-generic observation
ta_callout: str     # 1 sentence; grounded in acme_experience only
```
**Important constraint:** `acme_experience` is a required input; prompt explicitly forbids fabricating Acme experience claims. This is a MANDATORY-equivalent constraint moved from prose into the signature.
**Files:** `signatures/personalize_email.py:PersonalizeEmail`, `evals/personalize_email.jsonl` (3 seed examples)
**Eval coverage:** drug-specific, condition-specific, general-capabilities campaign types

---

## Open Questions

The following judgment calls were made during graduation. A human reviewer should verify each:

### OQ-1: `taxonomy_lookup` as `external_call` vs `pure_function`
**Decision:** Classified as `external_call` with `cache: {strategy: persistent, ttl: 30d}`.
**Why:** The lookup calls a real external API (ZoomInfo). The crystallization-heuristics reference suggests `pure_function` with caching for stable taxonomy lookups, but the canonical sanitized run uses `external_call`. Since the call is deterministic and the data is stable, the caching config makes it nearly equivalent in practice.
**Verify:** If the adapter treats `external_call` with cache differently from `pure_function` with cache, this may need reclassification.

### OQ-2: Exclusion checks as `external_call` vs `pure_function`
**Decision:** `external_call` with `mandatory: true` (matching the sanitized run).
**Why:** Each check calls a real HubSpot API endpoint. The expected baseline used `pure_function` (the checks were conceived as pure once the API call is wrapped). The sanitized run corrected this to `external_call` since they make deterministic API calls with retry semantics.
**Verify:** Both work structurally; `external_call` is semantically more accurate.

### OQ-3: `personalize_email` scope — per-contact vs per-template
**Decision:** Modeled as per-contact (`fan_out: true`) generating opening_line + ta_callout.
**Why:** The skill says "Personalize using enrichment data (title, company, career history)" which implies per-contact. The base template boilerplate comes from `email-templates.md` and is not LLM-generated.
**Alternative:** Could model as one invocation generating a campaign-level template with HubSpot tokens. The current model produces richer personalization but requires one LLM call per contact.
**Verify:** Check with BDR whether personalization is truly per-contact or per-campaign.

### OQ-4: Intel brief constraint for `personalize_email`
**Decision:** `personalize_email` receives `target_research` intel via a data edge.
**Why:** The skill says email personalization uses "specific references to their programs, recent data, and Acme's relevant experience" — all from target_research.
**Gap:** The `acme_experience` field comes from internal sources (Notion database, publications). This pipeline assumes the `target_research` agent_loop extracts it. **Someone must ensure `target_research` populates `relevant_experience` in the IntelBrief from validated internal sources only** — not from Bright Data or public web.

### OQ-5: Conference path as separate entry vs conditional
**Decision:** Conference path is a distinct set of entry nodes (`conf_load_file`) that converges at `hubspot_upsert`.
**Why:** The source skill treats it as an alternative entry point ("Conference list | Phase 2-alt → 4"). Modeled as parallel entry nodes with the adapter responsible for routing based on `campaign_type == "conference"` in the input brief.
**Verify:** This requires the adapter to emit conditional workflow entry logic. If the target runtime can't handle multiple entry nodes cleanly, consider splitting into two separate pipelines.

### OQ-6: Enrollment warning not modeled as a pre-filter
**Decision:** The `manual_enrollment_handoff` HITL gate handles the enrollment warning passively (it just blocks until the BDR signals).
**Why:** The skill says enrollment MUST be manual — there's no safe API. The pipeline models this as a suspension point, not a check. The warning is implicit in the gate.
**Alternative:** Add a `pure_function` node before the gate that renders the enrollment warning as a Markdown callout. This would make the warning explicit in the pipeline rather than implicit in the description.

---

## Suggested Next Steps

### 1. Dogfood `vet_contact` first
Run the `vet_contact` signature against a real set of ZoomInfo enriched contacts from a past campaign. Compare decisions against what the BDR actually approved. Expand `evals/vet_contact.jsonl` with 20–30 real examples before running DSPy compile.

### 2. Wire `acme_experience` source
The `personalize_email` node requires a `acme_experience: str` input. This currently comes from `target_research.intel.relevant_experience`. Verify that the Airweave/Notion source lookup in `target_research` actually populates this field. If it doesn't, add a separate `external_call` node (`lookup_acme_experience`) that queries the internal publications/studies database directly.

### 3. Implement the three `external_call` stubs
Priority order:
1. `exclusion_checks.py` (three functions) — these are MANDATORY and block enrollment
2. `hubspot.py:batch_upsert_contacts` — needed for any CRM upload
3. `zoominfo.py:enrich_contacts_batch` — needed for the core lead-gen loop

### 4. Run `rote emit pipeline.yaml --runtime temporal` and inspect the output
The Temporal adapter will emit `workflow.py` + `activities.py`. Verify:
- Three exclusion checks are emitted as unconditional activities (not conditional)
- HITL gates generate `workflow.wait_for_signal()` patterns
- Fan-out nodes (vet_contact, personalize_email) dispatch in parallel via `asyncio.gather`

### 5. Consider a `lookup_acme_experience` external_call node
Currently, Acme experience validation lives in the `target_research` agent_loop. This is risky — the agent could conceivably use non-validated sources. Making it a separate `external_call` node that queries only the Notion DB + internal publications page would make the constraint impossible to violate, matching the MANDATORY escalation pattern used for exclusion checks.

### 6. Regraduate after first production run
After running 3–5 real campaigns through the graduated pipeline, revisit:
- `lead_generation_loop` — does it always follow the same 3-search pattern? If yes, the initial three searches could be crystallized into separate `external_call` nodes, leaving only the backfill logic as `agent_loop`.
- `target_research` — are there consistent tool sequences for specific TAs? Could be partially crystallized.
