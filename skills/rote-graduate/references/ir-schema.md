# Pipeline IR Schema (`pipeline.yaml`)

Phase 5 of the graduator produces a `pipeline.yaml` file that describes
the entire graduated pipeline in a runtime-agnostic form. This file is
the reference for that YAML schema. It matches the Pydantic models in
`rote/ir.py`; when in doubt, that module is the authoritative source.

## Top-level structure

```yaml
name: bdr-campaign              # required, kebab-case, unique
version: "0.1.0"                # required, semver string
source_skill: ../../skill       # optional, path to source skill bundle
description: |                  # optional, multi-line prose
  End-to-end BDR outreach campaign workflow...

config:                         # optional, defaults apply if omitted
  schedule: null                # cron expression or null
  on_failure: notify_owner      # free-form adapter hook
  observability:
    traces: true
    eval_set_dir: ./evals/
  hitl:
    default_timeout: 7d

input:                          # required, pipeline input contract
  type: CampaignBrief
  required:
    - drug_brand
    - drug_generic
  optional:
    - job_focus

nodes: [ ... ]                  # required, list of Node objects
edges: [ ... ]                  # required, list of Edge objects
entry_nodes: [target_research, taxonomy_lookup]
exit_nodes: [manual_enrollment_handoff]
```

## Node object

Every node has a `kind` field, and the required fields depend on the
kind. The validator enforces kind-specific requirements.

### Common fields (all kinds)

```yaml
- id: taxonomy_lookup            # required, unique, snake_case
  kind: pure_function            # required, one of 5 kinds
  phase: "2"                     # optional, source skill phase (string)
  description: |                 # required, short prose
    Resolve ZoomInfo IDs for management levels...
  input:                         # optional, field→type mapping
    brief: CampaignBrief
  output: TaxonomyIds            # optional, type name or field mapping
  timeout: 5m                    # optional, duration string
  retry:                         # optional
    max: 3
    backoff: exponential         # linear | exponential | constant
    retry_on: [rate_limit, network]   # optional, note: NOT 'on' — YAML parses 'on' as boolean
  mandatory: false               # optional, default false
  constants:                     # optional, arbitrary key/value dict
    batch_size: 10
  cache:                         # optional
    strategy: persistent
    ttl: 30d
  fan_out: false                 # optional, if true node invoked per input element
```

### `pure_function` and `external_call`

**Required:** `impl` — path to the extracted Python function in
`"extracted/foo.py:bar_func"` format.

```yaml
- id: pre_enrollment_report
  kind: pure_function
  description: Render the pre-enrollment report as Markdown.
  impl: extracted/report.py:generate_pre_enrollment_report
  input:
    campaign_name: str
    vetted_count: int
    passed_contacts: list[HubSpotContact]
    exclusions: list[ExclusionRecord]
    template_ids: list[str]
  output: report_markdown str

- id: hubspot_upsert
  kind: external_call
  description: Batch upsert contacts to HubSpot (100 per call).
  impl: extracted/hubspot.py:batch_upsert_contacts
  input:
    contacts: list[VettedContact]
  output:
    upserted: list[HubSpotContact]
  constants:
    batch_size: 100
  retry:
    max: 5
    backoff: exponential
  timeout: 60s
```

### `llm_judge`

**Required:** at least one of `signature` (legacy) or `signature_spec`
(structured). **Strongly preferred: emit both.** The Temporal adapter
prefers `signature_spec` when present and falls back to the legacy
path; the Cloudflare adapter and any future non-Python target *require*
`signature_spec` because there's no shared Python module to import.

- **`signature`** — path to a typed Python signature class in
  `"signatures/foo.py:FooClass"` format. Used by the Temporal adapter
  and by humans iterating on the signature with DSPy / BAML. Always
  emit this for runtimes that share Python with the extracted modules.

- **`signature_spec`** — runtime-agnostic structured form: JSON Schema
  for input + output (the same schemas Pydantic emits via
  `model_json_schema()`), the prompt template, and the LLM client
  config. This is the cross-language source of truth — adapters
  derive Pydantic / Zod / Go types from the schemas as needed. See
  [`llm-judge-extraction.md`](llm-judge-extraction.md) for the full
  derivation procedure.

```yaml
- id: vet_contact
  kind: llm_judge
  description: Apply the BDR red-flags rubric to a single contact.
  signature: signatures/vet_contact.py:VetContact     # legacy path (Temporal)
  signature_spec:                                      # structured form (all runtimes)
    input_schema:
      type: object
      required: [contact, brief, intel]
      properties:
        contact: {$ref: "#/$defs/EnrichedContact"}
        brief: {$ref: "#/$defs/CampaignBrief"}
        intel: {$ref: "#/$defs/IntelBrief"}
      $defs:
        # ... full Pydantic model_json_schema() output
    output_schema:
      type: object
      required: [decision, relevance_evidence]
      properties:
        decision: {enum: [keep, discard]}
        tier:
          anyOf: [{enum: [ideal, strong, good]}, {type: "null"}]
        discard_reason:
          anyOf: [{enum: [indication_mismatch, msl_role, ...]}, {type: "null"}]
        relevance_evidence: {type: string}
    prompt: |
      Apply the BDR vetting rubric to this contact.
      Contact: {{ contact }}
      Brief: {{ brief }}
      Intel: {{ intel }}
      Return your decision via the structured output tool.
    client: anthropic           # 'anthropic' | 'openai'
    model: claude-sonnet-4-6    # optional; adapter chooses default if omitted
    temperature: 0.0            # optional
  input:
    contact: EnrichedContact
    brief: CampaignBrief
    intel: IntelBrief
  output:
    decision: VetDecision
    tier: ContactTier
    discard_reason: DiscardReason
    relevance_evidence: str
  constants:
    min_accuracy_score: 85
  eval_set: evals/vet_contact.jsonl
  fan_out: true                  # one invocation per contact
```

### `agent_loop`

**Required:** `tools` — list of tool names the agent may call inside
the loop.
**Optional:** `loop_body` — list of sub-node IDs invoked each iteration.
**Optional:** `termination` — condition + max iterations.

```yaml
- id: lead_generation_loop
  kind: agent_loop
  description: Iterative search-enrich-vet loop.
  input:
    brief: CampaignBrief
    intel: IntelBrief
    target_quota: int
  output:
    vetted_contacts: list[VettedContact]
  tools:
    - zoominfo_search_contacts
    - zoominfo_search_companies
  loop_body:
    - enrich_contact_batch
    - vet_contact
  termination:
    condition: vetted_count >= target_quota
    max_iterations: 10
  timeout: 15m
```

**Note on `loop_body`:** sub-nodes listed here must also exist as
top-level nodes in the `nodes:` list. They're referenced from the
loop_body but also emitted as standalone activities (so they can be
tested in isolation and reused elsewhere). The adapter excludes them
from top-level execution waves to prevent double-dispatch.

### `hitl_gate`

**Required:** `signal` — name of the signal the workflow waits for.
**Optional:** `notify` — how to alert the human reviewer.

```yaml
- id: contact_review_gate
  kind: hitl_gate
  description: Present the vetted contact table to the user for approval.
  input:
    vetted_contacts: list[VettedContact]
  output:
    approved_contacts: list[VettedContact]
  signal: contact_review_approved
  timeout: 7d
  notify:
    channel: slack
    target: "#bdr-reviews"
    message_template: |
      BDR campaign awaiting review: {drug_brand} for {condition_acronym}
```

**Constraint:** `mandatory: true` is not allowed on `agent_loop` nodes
(meaningless) but is allowed on every other kind. For HITL gates,
`mandatory` is implicit — you can't skip a gate the workflow is
waiting on.

## Edge object

```yaml
edges:
  - { from: target_research,         to: lead_generation_loop }
  - { from: lead_generation_loop,    to: contact_review_gate }
  - { from: contact_review_gate,     to: hubspot_upsert, on_signal: approved }
  - { from: exclusion_check_sequence, to: personalize_email, fan_out: true }
```

- **`from` / `to`:** node IDs (must exist in the `nodes:` list).
- **`on_signal`:** optional. If set, the edge only activates when the
  named signal fires. Used for edges exiting a `hitl_gate`.
- **`fan_out`:** optional. If true, the destination is invoked once per
  element of the source's output.

**Note:** `from` is a reserved word in Python, so the IR's Pydantic
model uses the alias `from_`. In YAML, always write `from:`.

## Validation rules the graduator must respect

The IR validator will reject any of these, so produce the YAML with
these constraints in mind:

1. **Node IDs are unique** within a pipeline.
2. **All `from`/`to` in `edges` reference real node IDs.**
3. **All `loop_body` entries reference real node IDs.**
4. **All `entry_nodes` and `exit_nodes` reference real node IDs.**
5. **Kind-specific required fields are present:**
   - `pure_function` / `external_call` → `impl`
   - `llm_judge` → `signature` (legacy path) or `signature_spec`
     (structured) — at least one. Strongly preferred: emit both.
   - `agent_loop` → `tools`
   - `hitl_gate` → `signal`. Cloudflare additionally requires the
     signal name to match `[A-Za-z0-9_-]+` (no dots, spaces, etc.) —
     this is enforced at adapter emission time.
6. **`mandatory: true` is not allowed on `agent_loop` nodes.**
7. **No YAML key named `on:`** — YAML 1.1 parses it as boolean
   `True`. The IR uses `retry_on:` instead.

## Duration strings

Used in `timeout:`, `default_timeout:`, `cache.ttl:`, etc.

| Suffix | Meaning |
|---|---|
| `ms` | milliseconds |
| `s` | seconds |
| `m` | minutes |
| `h` | hours |
| `d` | days |

No suffix → seconds. Examples: `5m`, `30s`, `7d`, `250ms`.

## Minimal skeleton for a new pipeline

When generating a `pipeline.yaml` from scratch, start with this skeleton
and fill it in:

```yaml
name: <skill-name>
version: "0.1.0"
source_skill: <relative path to skill>
description: |
  <one-paragraph summary of what the skill does>

config:
  on_failure: notify_owner

input:
  type: <BriefTypeName>
  required: []
  optional: []

nodes: []
edges: []
entry_nodes: []
exit_nodes: []
```

Then add nodes from the source skill's entry point outward, adding
edges as you go. Use `entry_nodes` for nodes with no inbound edges
(except from the pipeline input itself) and `exit_nodes` for terminal
nodes whose completion means the workflow is done.

## Cross-referencing the rubric

- Every node's `kind` decision is guided by `node-kinds.md`.
- Every `pure_function` / `external_call` node's `impl` and `constants`
  come from applying the patterns in `crystallization-heuristics.md`.
- Every `llm_judge` node's `signature` design is guided by
  `llm-judge-extraction.md`.
- Every `mandatory: true` flag comes from a Pattern 3 hit in the
  crystallization scan.
