# BDR Outreach — Compilation Report

**Source skill:** `examples/bdr-outreach/skill/SKILL.md`
**Compiled:** 2026-07-18
**Compiler:** rote-compile (Claude Sonnet 4.6)

---

## Summary metrics

| Metric | Value |
|---|---|
| Total nodes | 22 |
| `pure_function` | 9 |
| `external_call` | 6 |
| `llm_judge` | 2 |
| `agent_loop` | 2 |
| `hitl_gate` | 3 |
| Mandatory nodes | 3 |
| Fan-out nodes | 2 |
| Loop-body nodes | 2 |
| Extracted Python modules | 5 |
| Typed LLM-judge signatures | 2 |
| Eval seed cases | 12 (8 + 4) |

### HITL gates

| Gate | Signal | Timeout | Blocks |
|---|---|---|---|
| `contact_review_gate` | `contact_review_approved` | 7 days | `hubspot_upsert` |
| `conf_exclusion_review_gate` | `exclusion_review_confirmed` | 7 days | `conf_enrich_batch` |
| `manual_enrollment_handoff` | `bdr_enrollment_complete` | 14 days | — (terminal) |

### Estimated agent-turn cost reduction

| Path | Before (source skill) | After (compiled pipeline) | Reduction |
|---|---|---|---|
| Main campaign | 75–320 turns | ~40–105 turns (LLM nodes only) | ~60–70% |
| Conference path | 11–72 turns | ~6–60 turns | ~40–55% |

The main-campaign savings come almost entirely from eliminating agentic execution of deterministic steps: 9 `pure_function` nodes and 6 `external_call` nodes no longer consume LLM reasoning. The 3 MANDATORY exclusion checks are the highest-confidence savings — they were prose checklists; they are now code.

After compilation, LLM cost concentrates at two typed, evaluable nodes:
- **`vet_contact`**: 30–80 calls per run (one per enriched contact)
- **`personalize_email`**: 10–25 calls per run (one per cleared contact)

These are unavoidable; the input is genuinely unbounded (contact profiles vary). DSPy compilation against the eval seeds can compress them further once baseline accuracy is measured.

---

## Crystallization log

Every prose-to-code extraction from Phase 3, with source and destination.

### Module: `extracted/zoominfo.py`

#### `resolve_taxonomy_ids(brief)`

**Source:** `references/lead-generation.md`, Phase 2 setup prose — "Query ZoomInfo's Management Level for VP Level Exec and Director, Department for Medical & Health, and Industry IDs for pharmaceutical, biotech."

**Before (prose):**
> Make four parallel taxonomy lookup calls to ZoomInfo. Record the IDs for management levels VP Level Exec and Director, the department 'Medical & Health', and the industries 'pharmaceutical' and 'biotech'. These IDs will be reused on every search in the loop.

**After (code):** `resolve_taxonomy_ids()` returns a `TaxonomyIds` dict keyed by category. Constants lifted: `TAXONOMY_LOOKUPS` dict with the four query strings. Cached 30 days.

---

#### `enrich_contacts_batch(contacts, output_fields=None)`

**Source:** `references/lead-generation.md`, "Enrich in batches of 10" + `references/quality-and-vetting.md`, "Always request employmentHistory".

**Before (prose):**
> Enrich contacts in batches of 10 using the ZoomInfo Enrich API. Always include employmentHistory — vetting requires the full career trajectory. Extract the LinkedIn URL from the externalUrls array.

**After (code):** Batch size constant `ENRICH_BATCH_SIZE = 10` enforced at call site with `ValueError` before any API call. `STANDARD_OUTPUT_FIELDS` and `CONFERENCE_OUTPUT_FIELDS` field lists lifted from the prose. `extract_linkedin_url()` helper extracted verbatim.

---

### Module: `extracted/hubspot.py`

#### `batch_upsert_contacts(contacts)`

**Source:** `references/hubspot-operations.md`, "Batch upsert up to 100 contacts per call using the Contacts API endpoint."

**Before:** Prose limit of 100; no enforcement.
**After:** `UPSERT_BATCH_SIZE = 100`; function raises `ValueError` before API call if batch exceeds limit.

---

#### `create_campaign_list(campaign_name)` / `add_contacts_to_list(list_id, contacts)`

**Source:** `references/hubspot-operations.md`, list creation + membership steps.

**Before:** Two separate prose steps with no batch enforcement for list membership.
**After:** `ADD_TO_LIST_BATCH_SIZE = 250` lifted as constant; `add_contacts_to_list` raises `ValueError` on oversized input and returns `list_url`.

---

#### `upsert_sales_template(campaign_name, personalizations)`

**Source:** `references/hubspot-operations.md`, "POST to /sales/v3/templates with cookie auth."

**Before:** Prose instruction to use an internal endpoint with no error handling.
**After:** Stub with the correct endpoint documented; cookie auth requirement captured in docstring. Regime 1 — user fills in the actual cookie auth mechanism.

---

### Module: `extracted/exclusion_checks.py` ← highest-value extraction

These three were MANDATORY prose checklists. Moving them to code makes them impossible to accidentally skip, condition, or reorder.

#### `check_do_not_contact(contacts, dnc_list_id)`

**Source:** `references/hubspot-operations.md`, Phase 5 — "MANDATORY: Check the BDR do not contact list before enrolling any contact."

**Before:**
> Before enrolling any contact in a sequence, check if they appear on the "BDR do not contact" list in HubSpot. Skip any contacts that appear on this list. This check is MANDATORY and cannot be omitted.

**After:** `DNC_LIST_QUERY = "BDR do not contact"` lifted as constant. Function signature forces the caller to pass all contacts through; cannot be conditionally skipped. Returns separate `passed` and `excluded` lists so the exclusion count flows into the pre-enrollment report.

---

#### `check_recently_emailed(contacts)`

**Source:** `references/hubspot-operations.md`, Phase 5 — "MANDATORY: Do not email contacts who received an outbound email in the last 30 days."

**Before:**
> Check if each contact was emailed in the last 30 days. Use the HubSpot Engagements API, filtering for OUTBOUND emails. Skip any contacts who received an outbound email within this window.

**After:** `RECENT_EMAIL_DAYS = 30` lifted as constant. Direction filter `OUTBOUND` lifted as constant. MANDATORY docstring marker.

---

#### `check_active_sequence(contacts)`

**Source:** `references/hubspot-operations.md`, Phase 5 — "MANDATORY: A contact can only be enrolled in one active sequence at a time."

**Before:** Prose rule with no enforcement mechanism.
**After:** Codified as pure function that checks enrollment status; returns `excluded` list for the report. MANDATORY marker.

---

### Module: `extracted/report.py` ← working implementations

Both report functions are pure string formatting with all inputs available from the skill. Implemented fully (not stubs).

#### `build_contact_table(vetted_contacts, drug_name, condition, discard_summary, total_searched)`

**Source:** `references/lead-generation.md`, lines 124–133 — table column spec (Name, Title, Company, Email, Tier, Evidence) + sort order (ideal → strong → good).

**After:** Fully working. Sorts by tier enum order, renders Markdown table with discard summary header. No external deps.

---

#### `generate_pre_enrollment_report(campaign_name, vetted_count, passed_contacts, exclusions, template_ids)`

**Source:** `references/hubspot-operations.md`, lines 86–103 — the exact Markdown template for the pre-enrollment report, including section headings and field layout.

**After:** Fully working. Template lifted verbatim from the skill prose. All fields are pure data from upstream nodes.

---

### Module: `extracted/conference.py`

#### `is_pharma(company_name) → bool` ← working implementation

**Source:** `references/conference-enrichment.md`, lines 52–85 — the Python function literally embedded in the skill's own reference.

**After:** Lifted verbatim. Exclude keywords checked first; ambiguous names default to `False` (excluded). `PHARMA_INCLUDE_KEYWORDS` and `PHARMA_EXCLUDE_KEYWORDS` lifted as module-level constants.

---

#### `filter_pharma_contacts(raw_contacts)`

**Source:** `references/conference-enrichment.md` — "For each attendee, run is_pharma(company_name)."

**After:** Fully working. Applies `is_pharma()` per contact; returns `(pharma, excluded)` tuple. No external deps.

---

#### `load_attendee_file(file_path)` / `build_conference_xlsx(enriched_contacts, conference_name)`

**Source:** `references/conference-enrichment.md` — CSV/XLSX load + formatting spec.

**After:** Stubs (Regime 1 — no probe context). `build_conference_xlsx` has all formatting constants: `HEADER_FILL_COLOR = "1F4E79"`, `ALT_ROW_FILL_COLOR = "EBF3FB"`, `COLUMN_WIDTHS`, `FONT_NAME = "Arial"`. Column order and filename format documented in docstrings.

---

### LLM-judge signatures

#### `signatures/vet_contact.py:VetContact`

Pre-filter at `MIN_ACCURACY_SCORE = 85` (lifted from `references/quality-and-vetting.md`) runs before any LLM call. Hard thresholds (accuracy, email presence) are impossible to soften via prompt drift.

`DiscardReason` enum: 10 discrete values covering all red flags in the skill rubric. Notable additions relative to previous runs: `translational`, `ops_strategy`, `program_management` as separate reasons (the rubric lists them separately in `references/quality-and-vetting.md`).

Eval seeds: 8 cases covering all 10 discard reasons, 3 tier levels, and both pre-filter paths.

---

#### `signatures/personalize_email.py:PersonalizeEmail`

Campaign-type-aware prompt: drug-specific allows naming the drug; condition-specific and general-capabilities must not. This constraint is **in the prompt**, not in a pre-filter — it's a genuine judgment call where the LLM must apply the rule per-contact. The eval seeds verify both hard constraints (assert_not_contains Dupixent for condition-specific) and soft quality constraints (assert_contains specific program names for ideal-tier contacts).

`acme_experience` field documented as "all claims must come only from this field — never fabricated." Enforced by prompt and tested in eval seeds.

---

## Open questions

These are judgment calls I made that a reviewer should verify before production use.

### 1. `taxonomy_lookup` as `pure_function` vs `external_call`

**What I chose:** `pure_function` with a 30-day persistent cache.

**Why:** The taxonomy IDs (VP Level Exec, Director, Medical & Health, pharmaceutical, biotech) are described in the skill as stable constants looked up once per campaign. The 30-day TTL assumes they don't change frequently.

**Verify:** If ZoomInfo updates its taxonomy IDs more frequently than once a month, or if a campaign's drug/condition would require different taxonomy IDs than the hardcoded ones, this should be reclassified as `external_call` with per-run cache-busting.

---

### 2. `exclusion_check_*` as `pure_function` vs `external_call`

**What I chose:** `pure_function`.

**Why:** Consistent with both reference runs and the crystallization heuristic that classifies "MANDATORY checks" as `pure_function` — the extracted functions are deterministic from the caller's perspective. The HubSpot API calls are implementation details inside the stub.

**Verify:** If you want the runtime to manage retries and timeouts at the node level for these checks, reclassify them as `external_call` and add `mcp:` bindings pointing to the HubSpot engagement history and list membership APIs. The MANDATORY invariant holds either way.

---

### 3. Conference path convergence at `hubspot_upsert`

**What I chose:** A single convergence edge `conf_enrich_batch → hubspot_upsert`, with `hubspot_upsert.inputs.contacts` binding only the main path (`contact_review_gate.output.approved_contacts`).

**Why:** The 2026-07-03 compiled run used this same pattern, noting the edge "signals convergence" for the adapter. The conference path is an alternate entry that should produce the same CRM upload behavior without going through review.

**Verify:** This is the most ambiguous DAG decision. If the runtime treats `hubspot_upsert` as an AND-join (waits for both `contact_review_gate` AND `conf_enrich_batch`), it will deadlock whenever only one path runs. The runtime must detect that these are alternate paths, not parallel ones. Confirm the target adapter handles this correctly before running both paths in the same execution.

---

### 4. `vet_contact` `fan_out: true` + `loop_body` combination

**What I chose:** Both `fan_out: true` and `loop_body: [enrich_contact_batch, vet_contact]` on `lead_generation_loop`.

**Why:** Inside the lead gen loop, vetting is applied per contact (fan-out). At the lead gen loop's output level, all vet decisions are aggregated into `vetted_contacts`. The combination means: the runtime calls `vet_contact` once per element of `enrich_contact_batch.output.enriched`, within each iteration of `lead_generation_loop`.

**Verify:** Not all adapters implement fan-out + loop-body combinations identically. Test the emitted code to confirm per-element dispatch happens inside the loop iteration, not over the entire run's contact set.

---

### 5. `create_sales_template` MCP tool name

**What I chose:** `mcp: {server: hubspot, tool: hubspot_create_sales_template}`

**Why:** The skill mentions "POST to /sales/v3/templates" but doesn't name an MCP tool explicitly.

**Verify:** The actual HubSpot MCP server may expose this differently — possibly as `hubspot_upsert_sales_template` or `hubspot_create_email_template`. Check the server's tool manifest with `rote mcp list hubspot` and update the `mcp:` binding accordingly.

---

### 6. `pre_enrollment_report` inputs — `vetted_count` bound to `total_searched`

**What I chose:** `vetted_count: lead_generation_loop.output.total_searched`

**Why:** The 2026-07-03 run uses this same binding. The skill's report template asks for "X contacts vetted across Y searches" — using `total_searched` for the vetted count is arguably a naming mismatch (total_searched counts raw contacts run through ZoomInfo, not the final vetted set).

**Verify:** If `generate_pre_enrollment_report` should show the count of contacts that *passed* vetting (before exclusions), bind to a count derived from `lead_generation_loop.output.vetted_contacts` instead. The report template in `hubspot-operations.md` lines 86–103 should clarify the intent.

---

## Suggested next steps

### 1. Validate the IR and run `rote emit`

```sh
rote emit /tmp/rote-compile-hynlit85/pipeline.yaml --runtime dbos --out /tmp/bdr-dbos
```

This will surface any IR validation errors (charset violations, missing `impl:` targets, undefined node references) before any production work begins.

---

### 2. Fill the MANDATORY exclusion check stubs first

`exclusion_check_dnc`, `exclusion_check_recent`, `exclusion_check_sequence` are the highest-safety nodes. Fill them in with real HubSpot SDK calls before testing anything else. A compilation that still has stubs here is not production-safe — these three checks are the difference between a professional campaign and a compliance incident.

---

### 3. Run eval seeds against a real model

```sh
rote eval pipeline.yaml --run --node vet_contact
rote eval pipeline.yaml --run --node personalize_email
```

The 8 `vet_contact` seeds cover all 10 discard reasons and 3 tiers; they will tell you quickly if the prompt needs tuning. Pay special attention to `indication_mismatch` cases — these require the model to reason about TA overlap, which is the most context-dependent judgment in the rubric.

---

### 4. Test `is_pharma()` on a real conference list

`filter_pharma_contacts` is a working implementation and the most likely function to need tuning. Run it against a real ISPOR or ASH conference attendee list and measure precision/recall before using it to gate enrichment spend.

---

### 5. Wire MCP auth before any production run

```sh
rote mcp login hubspot
rote mcp login zoominfo
```

The park-on-auth mechanism will hold the workflow until credentials are configured. Do this setup before the first real run, not after watching a workflow park in production.

---

### 6. Recompile after the first production run

The `source.section` provenance on every node enables `rote compile --update` for incremental changes. After running one real BDR campaign, you will likely learn:

- Which `DiscardReason` values need to be split or merged (the `ops_strategy` / `program_management` split vs. previous runs' `it_systems` / `clinical_researcher` split — real data will tell you which taxonomy is right)
- Whether `target_research` genuinely needs all 5 tools or whether 2-3 are sufficient
- Whether the 30-day taxonomy cache TTL is too long or too short

Commit those learnings back to the skill prose and recompile from the new source — that's the rote flywheel.
