# LLM Judge Extraction

Phase 4 of the graduator turns every `llm_judge` node's fuzzy prose
rubric into a typed signature with bounded inputs and outputs. This
file is the reference for that extraction.

The premise is simple: if the source skill's rubric defines a
classification (red flags, tiers, decision categories), that
classification has a natural schema — and *running it as an unbounded
LLM prompt throws away all the structure*. A typed signature recovers
the structure, makes the step regression-testable, and lets downstream
nodes consume a predictable shape instead of parsing free text.

## What a typed signature looks like

A typed signature is a Python class (or BAML function, or DSPy
Signature) with three parts:

1. **Input model** — a Pydantic BaseModel with every field the LLM
   needs, sourced from upstream nodes.
2. **Output model** — a Pydantic BaseModel with enum-bounded fields
   wherever the rubric implies a discrete choice, plus any structured
   rationale.
3. **Forward method** — the call site that dispatches to the LLM. For
   v0 this is a stub; downstream work fills it in with DSPy or BAML.

The BDR example `signatures/vet_contact.py` is the canonical reference
implementation.

## The extraction procedure

### Step 1 — Read the source rubric

Find every piece of prose in the source skill that describes how to
classify the input. For BDR's `vet_contact`, this is
`references/quality-and-vetting.md`:

- High-signal indicators (boosts)
- Red flags (discard)
- The core test ("would this person commission an RWE study?")
- Tier definitions (ideal / strong / good)
- Numeric thresholds (accuracy ≥ 85)

### Step 2 — Enumerate the decision space

Ask: what are the possible outputs? For `vet_contact`:

| Dimension | Values |
|---|---|
| Decision | `keep`, `discard` |
| Tier (if keep) | `ideal`, `strong`, `good` |
| Reason (if discard) | `indication_mismatch`, `msl_role`, `biomarker_discovery`, `translational`, `sales_commercial`, `ops_strategy`, `program_management`, `low_accuracy`, `no_valid_email`, `other` |
| Evidence | free-form string (1-2 sentences) |

Anything with a small enumerable value space becomes an **enum**. Free
text is only used for the *evidence* or *explanation* field, never for
the core decision.

**Rule of thumb:** if you can write the values in a table like the one
above, it's an enum. If the values would span a paragraph, it's free
text.

### Step 3 — Design the input model

Walk the IR backward from the node. The input model must contain every
field the LLM needs to make the decision. For `vet_contact`:

```python
class VetContactInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contact: EnrichedContact   # from enrich_contact_batch upstream
    brief: CampaignBrief        # from pipeline input
    intel: IntelBrief           # from target_research upstream
```

**`extra="forbid"`** prevents accidental field additions that would
silently break downstream reads. Strict-by-default.

### Step 4 — Design the output model

```python
class VetDecision(str, Enum):
    KEEP = "keep"
    DISCARD = "discard"

class ContactTier(str, Enum):
    IDEAL = "ideal"
    STRONG = "strong"
    GOOD = "good"

class DiscardReason(str, Enum):
    INDICATION_MISMATCH = "indication_mismatch"
    MSL_ROLE = "msl_role"
    BIOMARKER_DISCOVERY = "biomarker_discovery"
    # ... one per rubric category

class VetContactOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: VetDecision
    tier: ContactTier | None = None           # only when decision == keep
    discard_reason: DiscardReason | None = None   # only when decision == discard
    relevance_evidence: str
```

**Optional fields with `None` defaults** are how you express "this
field only applies in some decision branches" without making the
schema conditional. Downstream consumers handle the None case.

### Step 5 — Lift hard thresholds into a pre-filter

If the rubric contains any numeric threshold or enum check that can be
evaluated without the LLM, put it in the `forward()` method as a
pre-filter that short-circuits before calling the model.

From `signatures/vet_contact.py`:

```python
MIN_ACCURACY_SCORE: int = 85

class VetContact:
    async def forward(self, inputs: VetContactInput) -> VetContactOutput:
        if inputs.contact.accuracy_score < MIN_ACCURACY_SCORE:
            return VetContactOutput(
                decision=VetDecision.DISCARD,
                discard_reason=DiscardReason.LOW_ACCURACY,
                relevance_evidence=(
                    f"Accuracy score {inputs.contact.accuracy_score} "
                    f"below threshold {MIN_ACCURACY_SCORE}."
                ),
            )
        # ... dispatch to LLM for fuzzy cases
```

**Why this matters:** the pre-filter saves tokens on the obvious cases
(probably 20-40% of real contacts) and guarantees the hard rule cannot
drift. The LLM can never "forget" the accuracy threshold because it
never sees those contacts.

### Step 6 — Scaffold a seed eval set

For every `llm_judge` node, create a seed evals file at the path
referenced by the node's `eval_set:` field. Each line is one test case.

Harvest examples directly from the source rubric. For every
discard_reason enum value, construct at least one input that should
produce that reason.

For BDR `evals/vet_contact.jsonl`:

```jsonl
{"name": "msl_role_discard", "input": {"contact": {"job_title": "Medical Science Liaison, Respiratory", "employment_history": [{"title": "MSL, Oncology"}], "accuracy_score": 95}, "brief": {"therapeutic_area": "respiratory"}}, "expected": {"decision": "discard", "discard_reason": "msl_role"}}
{"name": "biomarker_discard", "input": {"contact": {"job_title": "Director, Biomarker Sciences", "accuracy_score": 92}}, "expected": {"decision": "discard", "discard_reason": "biomarker_discovery"}}
{"name": "low_accuracy_discard", "input": {"contact": {"job_title": "Sr. Director RWE", "accuracy_score": 70}}, "expected": {"decision": "discard", "discard_reason": "low_accuracy"}}
{"name": "ideal_coe", "input": {"contact": {"job_title": "Sr. Director, Real World Evidence CoE", "employment_history": [...]}}, "expected": {"decision": "keep", "tier": "ideal"}}
```

These seed examples are not a full eval suite — they're the starting
point a human can expand as they encounter real cases. But having them
in the repo means the downstream DSPy/BAML compile step has a regression
baseline to optimize against.

## Common patterns

### Pattern: "discard categories" rubric

The source skill has an explicit list of categories to reject. Each
becomes an enum member of a `Reason` enum, and the eval set has one
example per category.

### Pattern: "tier the keepers" rubric

The source skill defines tiers (ideal / strong / good) with criteria
for each. Each tier becomes an enum member of a `Tier` field, which is
optional (only set when the decision is to keep).

### Pattern: "one judge, fan out"

If the source skill applies the same judgment to a list of inputs
(vet 50 contacts, personalize 10 emails), set `fan_out: true` on the
node in the IR. The adapter will invoke the signature once per input
element in parallel. The signature itself handles one input at a time.

### Pattern: "cheap pre-filter + expensive LLM"

Almost every `llm_judge` has at least one numeric or enum constraint
hiding in the rubric. Always check.

## What does NOT belong in a signature

- **Exploratory work.** If the "classification" requires the agent to
  decide which external sources to consult, it's an `agent_loop`, not
  an `llm_judge`.
- **Unbounded generation.** If the output is a paragraph of free prose
  with no schema, reconsider whether it's the right kind. Most
  "generate a paragraph" tasks can be narrowed to a few structured
  fields (opening line + TA callout + CTA, for example).
- **Multi-step reasoning across many inputs.** Signatures take one
  bounded input and return one bounded output. If the step needs to
  reason across a batch, use `fan_out: true` so each input is its
  own invocation.

## The output of Phase 4

For each `llm_judge` node:

1. A file at `signatures/<node_name>.py` with:
   - The input model
   - The output model
   - The enum definitions
   - The signature class with a stub `forward()` method (pre-filter
     logic included; LLM dispatch raises `NotImplementedError`)
2. A file at `evals/<node_name>.jsonl` with 3–10 seed examples, one
   per distinct decision path.

Record every signature extracted in the Phase 7 graduation report with
the source rubric location and a one-line summary of the output schema.
This is the audit trail for the human reviewer.
