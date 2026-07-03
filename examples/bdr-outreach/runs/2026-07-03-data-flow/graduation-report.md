# Graduation Report — BDR Outreach Campaign

Skill: `examples/bdr-outreach/skill/SKILL.md`
Graduated: 2026-07-03
Graduator: rote-graduate (manual, via Claude Code)

---

## Summary Metrics

| Metric | Value |
|---|---|
| Total nodes | 22 |
| Agent loop nodes | 2 |
| LLM judge nodes | 2 |
| Pure function nodes | 9 |
| External call nodes | 6 |
| HITL gate nodes | 3 |
| MANDATORY nodes | 3 |
| HITL gates | 3 |
| Edges | 23 |
| Entry nodes | 3 (target\_research, taxonomy\_lookup, conf\_load\_file) |
| Exit nodes | 2 (manual\_enrollment\_handoff, conf\_build\_xlsx) |
| Extracted Python modules | 7 (zoominfo, hubspot, exclusion\_checks, report, taxonomy, conference, signatures/vet\_contact, signatures/personalize\_email) |
| Eval seed examples | 12 (8 for vet\_contact, 4 for personalize\_email) |

**Estimated token cost reduction per run:** ~60–70%. The previous skill ran the full 7-phase procedure as a continuous agent loop with 30–50 tool calls. The graduated pipeline codifies all fixed procedures into deterministic code; only `target_research` (genuine exploration), `lead_generation_loop` (loop control + ZoomInfo query strategy), `vet_contact` (rubric-bounded judgment per contact), and `personalize_email` (per-contact copy generation) remain as LLM steps. All CRM operations, exclusion checks, taxonomy resolution, table rendering, and conference XLSX generation are now zero-token.

**HITL gates and what they block:**

| Gate | Signal | Blocks | Default Timeout |
|---|---|---|---|
| `contact_review_gate` | `contact_review_approved` | CRM upload, all downstream | 7d |
| `conf_exclusion_review_gate` | `exclusion_review_confirmed` | ZoomInfo enrichment (costly per-contact) | 7d |
| `manual_enrollment_handoff` | `bdr_enrollment_complete` | Nothing (terminal) | 14d |

---

## Node Classification Decisions

| Node | Kind | Justification |
|---|---|---|
| target\_research | agent\_loop | Genuinely exploratory — multi-source research (Bright Data, ClinicalTrials.gov, Airweave, Salesforce), output varies per company and drug |
| taxonomy\_lookup | pure\_function | Four deterministic API GETs to ZoomInfo Lookup API; IDs are stable for 30+ days; cached |
| lead\_generation\_loop | agent\_loop | Loop control is exploratory — agent decides when to backfill, which search strategy to try next, when quota is met |
| enrich\_contact\_batch | external\_call | Deterministic API call with fixed output fields and batch size=10; needs retry semantics |
| vet\_contact | llm\_judge | Rubric-bounded classification with typed enum output; pre-filters (accuracy, email) are pure code in `forward()` |
| build\_contact\_table | pure\_function | Fixed Markdown template from lead-generation.md lines 124–133 |
| contact\_review\_gate | hitl\_gate | Explicit "present to user" phase in the skill |
| hubspot\_upsert | external\_call | Deterministic POST to HubSpot Contacts API, batch size=100 |
| hubspot\_create\_list | external\_call | Deterministic API call, returns list\_id |
| hubspot\_add\_to\_list | external\_call | Deterministic API call, batch size=250 |
| exclusion\_check\_dnc | pure\_function | Fixed query against "BDR do not contact" list; MANDATORY |
| exclusion\_check\_recent | pure\_function | Fixed date-range query (RECENT\_EMAIL\_DAYS=30, OUTBOUND direction); MANDATORY |
| exclusion\_check\_sequence | pure\_function | Fixed API check for active sequence membership; MANDATORY |
| personalize\_email | llm\_judge | Ambiguous copy generation against implicit style rubric; campaign\_type controls content rules |
| create\_sales\_template | external\_call | Deterministic POST to HubSpot internal Sales Content API (requires cookies) |
| pre\_enrollment\_report | pure\_function | Fixed Markdown template from hubspot-operations.md lines 86–103 |
| manual\_enrollment\_handoff | hitl\_gate | Explicit "BDR must enroll manually" gate from Phase 7 |
| conf\_load\_file | pure\_function | Deterministic file parse (CSV/XLSX → normalized dicts) |
| conf\_pharma\_filter | pure\_function | Deterministic keyword classifier (`is_pharma()`, lifted verbatim) |
| conf\_exclusion\_review\_gate | hitl\_gate | Required cost-gate before ZoomInfo enrichment (per-contact billing) |
| conf\_enrich\_batch | external\_call | Same ZoomInfo enrichment node as main path, different entry point |
| conf\_build\_xlsx | pure\_function | Fixed XLSX formatting from conference-enrichment.md lines 148–174 |

---

## Crystallization Log

Every extraction below moves a procedure from prose (or implicit skill behavior) into deterministic, version-controlled Python.

### 1. `is_pharma()` — lifted verbatim

**Source:** `references/conference-enrichment.md` lines 52–85  
**Before:** A paragraph in the skill describing keyword matching logic ("check if any of these 30 keywords appear in the company name...")  
**After:** `extracted/conference.py:is_pharma(company_name: str) -> bool` — exact include/exclude keyword lists as module constants, logic identical to the prose description.

```python
# Before (prose in conference-enrichment.md):
# "Check if the company name contains any pharma/biotech keywords like
#  'pharma', 'biopharm', 'therapeutics', 'biologics'... etc.
#  Exclude if it contains 'consulting', 'university', 'hospital'..."

# After (extracted/conference.py):
PHARMA_INCLUDE_KEYWORDS = ("pharma", "biopharm", "therapeutics", ...)
PHARMA_EXCLUDE_KEYWORDS = ("consulting", "university", "hospital", ...)

def is_pharma(company_name: str) -> bool:
    name = company_name.lower()
    for kw in PHARMA_EXCLUDE_KEYWORDS:
        if kw in name: return False
    for kw in PHARMA_INCLUDE_KEYWORDS:
        if kw in name: return True
    return False
```

**Value:** The keyword lists are now versionable, testable, and can be extended without re-graduating.

---

### 2. Three MANDATORY exclusion checks — prose → `mandatory: true` pure_function nodes

**Source:** `references/hubspot-operations.md` lines 42–73  
**Before:** The skill uses ALL-CAPS "MANDATORY" in English prose three times. These checks were enforced only by the agent following instructions.  
**After:** Three `pure_function` nodes with `mandatory: true` in the IR. The rote IR validator rejects any attempt to mark these conditional. Adapters emit them as unconditional activities.

```yaml
# Before (prose):
# "MANDATORY: Check if any contacts are on the 'BDR do not contact' list..."
# "MANDATORY: Check if any contacts were emailed in the last 30 days..."
# "MANDATORY: Check if any contacts are in an active sequence..."

# After (pipeline.yaml):
- id: exclusion_check_dnc
  kind: pure_function
  mandatory: true
  ...
```

**Value:** Impossible to accidentally skip. Moving "MANDATORY" from English prose to a validator-enforced IR field is the single highest-value crystallization in this skill.

---

### 3. Batch size constants — promoted to module-level

**Source:** `references/lead-generation.md` lines 28–30, `references/hubspot-operations.md` lines 78–85  
**Before:** Inline numbers embedded in prose: "batch of 10", "100 contacts per call", "250 contacts per add".  
**After:** Constants in extracted modules and node `constants:` blocks:

```python
ENRICH_BATCH_SIZE = 10      # extracted/zoominfo.py
UPSERT_BATCH_SIZE = 100     # extracted/hubspot.py
ADD_TO_LIST_BATCH_SIZE = 250  # extracted/hubspot.py
```

---

### 4. TAXONOMY_LOOKUPS — four API calls promoted to a single cached pure_function

**Source:** `references/lead-generation.md` lines 19–26  
**Before:** The skill described four parallel `zoominfo_lookup` tool calls (management levels, pharma industry, biotech industry, medical department) in every run.  
**After:** `extracted/taxonomy.py:resolve_taxonomy_ids()` with a 30-day persistent cache. The taxonomy IDs don't change between campaigns or months; caching eliminates ~4 API calls per run.

---

### 5. Contact table template — fixed Markdown format

**Source:** `references/lead-generation.md` lines 124–133  
**Before:** Implicit table format described in prose; the agent would render it differently on each run.  
**After:** `extracted/report.py:build_contact_table()` — fixed column order, tier sorting (ideal → strong → good), and discard summary line. Output is deterministic given the same input.

---

### 6. Pre-enrollment report template — fixed Markdown

**Source:** `references/hubspot-operations.md` lines 86–103  
**Before:** A template described in prose that the agent would fill in from memory.  
**After:** `extracted/report.py:generate_pre_enrollment_report()` — fixed section headings, counts, tier breakdown, and "Next step" copy. Includes the critical manual enrollment note ("API enrollment sends emails immediately...").

---

### 7. XLSX formatting constants — openpyxl-ready

**Source:** `references/conference-enrichment.md` lines 148–174  
**Before:** Formatting instructions in prose ("dark navy header", "alternating light blue rows", "Arial font").  
**After:** `extracted/conference.py` constants:
```python
HEADER_FILL_COLOR = "1F4E79"   # dark navy
ALT_ROW_FILL_COLOR = "EBF3FB"  # light blue
COLUMN_WIDTHS = {"Company": 30, "First Name": 18, ...}
```

---

### 8. Pre-filter logic moved into `vet_contact.forward()`

**Source:** `references/quality-and-vetting.md` lines 18–21  
**Before:** The skill told the LLM to check accuracy score before vetting. If the LLM missed this instruction, it would waste tokens on a vet call for an unvetable contact.  
**After:** `signatures/vet_contact.py:VetContact.forward()` runs two deterministic checks before dispatching to the LLM:
```python
def forward(self, contact, brief, intel):
    if contact.accuracy_score < MIN_ACCURACY_SCORE:
        return VetContactOutput(decision="discard", discard_reason="low_accuracy", ...)
    if not contact.email:
        return VetContactOutput(decision="discard", discard_reason="no_valid_email", ...)
    raise NotImplementedError("replace with LLM dispatch")
```

---

## Open Questions

1. **taxonomy\_lookup classified as `pure_function` (not `external_call`):** The ZoomInfo Lookup API calls are deterministic and cached 30 days, so the adapter treats this node like a function rather than a retryable external service. This matches the expected/pipeline.yaml baseline. If the ZoomInfo Lookup API has rate limits that require retry semantics separate from the cache, reclassify as `external_call` and add `retry: {max: 3}`.

2. **Conference path convergence at `hubspot_upsert`:** The `hubspot_upsert` node has `inputs: contacts: contact_review_gate.output.approved_contacts` (covers the main path). The conference path reaches `hubspot_upsert` via the edge `conf_enrich_batch → hubspot_upsert`. Adapters handle this OR-convergence at the edge level — the binding tells them which input to use for the main path; the conference edge tells them to run the node when `conf_enrich_batch` completes too. The expected/pipeline.yaml explicitly excluded the conference path and left this unresolved. **Verify with the Temporal and Cloudflare adapters that this two-source convergence works before shipping.**

3. **`acme_experience` input to `personalize_email`:** This 2-3 sentence TA experience summary is in `pipeline.input.acme_experience` (optional). The BDR must supply it when launching the pipeline. If it's null, the email personalization will have a weaker `ta_callout`. Consider making it required, or defaulting to a boilerplate string inside `personalize_email.forward()` rather than leaving the LLM without context.

4. **`hubspot_create_list` already adds contacts:** Looking at the references, `create_campaign_list` calls the HubSpot API to create the list AND may batch-add contacts in the same call. The `hubspot_add_to_list` node exists as a separate step for the `ADD_TO_LIST_BATCH_SIZE=250` batching. Verify whether `batch_upsert_contacts` + `create_campaign_list` + `add_contacts_to_list` is the right three-step sequence or if it can be collapsed to two steps.

5. **`conf_build_xlsx` exit node placement:** The conference path ends at `conf_build_xlsx` (XLSX for conference coordinators) AND feeds into `hubspot_upsert` for CRM upload. If the conference path always continues through the full CRM path, `conf_build_xlsx` should not be an exit node — it's an intermediate parallel output. Clarify with the BDR team whether XLSX delivery ends the conference sub-flow or whether CRM upload is always expected.

6. **`manual_enrollment_handoff` 14-day timeout:** The main path gate waits 14 days; the two review gates wait 7 days. If the BDR doesn't respond in 14 days, the `on_failure: notify_owner` config fires. Consider whether a shorter timeout (3–5 days) is more appropriate for the handoff gate, since by Phase 7 all the expensive upstream work is done.

7. **`personalize_email` model selection:** Currently `claude-sonnet-4-6`. This is correct for most campaigns. For very high-value campaigns (>50 ideal-tier contacts), consider Opus 4.8 for the email personalization pass — the quality difference at the top tier is measurable. Add a `campaign_tier` input flag or a model override config to the node to make this switchable.

---

## Suggested Next Steps

### Immediate (before production)

1. **Fill in the `NotImplementedError` stubs** in all `extracted/*.py` modules. Each function has a docstring explaining the underlying API call. Start with `extracted/exclusion_checks.py` — these are the MANDATORY nodes and need to work correctly first.

2. **Run `rote emit pipeline.yaml --runtime temporal --out /tmp/bdr-temporal`** and review the emitted `workflow.py` + `activities.py`. The data-flow threading for the three-node exclusion chain (dnc → recent → sequence) is the most complex part of the emission.

3. **Validate conference path convergence** with the Temporal adapter using a `WorkflowEnvironment.start_time_skipping()` test that exercises the `conf_enrich_batch → hubspot_upsert` edge.

### First dogfood run

4. **Run a live graduation on a new skill** using the graduated pipeline as the execution engine. The lead generation loop and vet_contact node are the most expensive steps — compare per-contact LLM call count against the pre-graduation agent run.

5. **Eval the vet\_contact signature** against `evals/vet_contact.jsonl`. The 8 seed examples cover all 8 discard reasons and two keep tiers. After 5+ real runs, expand the eval set with borderline cases (VP Medical Affairs without an RWE focus, Director titles at CROs, etc.).

### After first production run

6. **Re-examine `lead_generation_loop` termination** — the `max_iterations: 10` limit is conservative. The real BDR campaigns typically converge in 3–5 iterations for a `target_quota` of 25–40. Lower to 6 after observing real runs.

7. **Promote `build_contact_table` to the expected/pipeline.yaml baseline.** The expected baseline doesn't include this node; this graduation adds it as a clean separation between the vetting loop output and the HITL gate presentation. It's a clear improvement — add it to the regression snapshot.

8. **Consider splitting the conference path** into a separate `pipeline.yaml` (as the expected/pipeline.yaml suggests). The convergence at `hubspot_upsert` works in theory, but two separate pipelines (one per entry point) would be simpler to test, monitor, and maintain independently. The graduation included both paths to be complete; the production architecture may prefer the split.
