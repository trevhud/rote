# Compile Report: rote-compile (self-compilation)

**Source skill:** `../../../../../skills/rote-compile`
**Compiler version:** rote-compile skill (self-referential dogfood run)
**Date:** 2026-09-04

---

## Summary metrics

| Metric | Value |
|--------|-------|
| Total nodes | 12 |
| Codifiable (pure_function + external_call) | 6 of 11 non-gate nodes |
| Codifiable % | **55%** |
| LLM-judge nodes | 3 |
| Agent-loop nodes | 3 |
| HITL gates | 0 |

### Node breakdown by kind

| Kind | Count | Nodes |
|------|-------|-------|
| `pure_function` | 4 | `detect_entry_point`, `write_eval_sidecar`, `validate_ir`, `write_compile_report` |
| `external_call` | 2 | `read_skill_bundle`, `invoke_adapter` |
| `llm_judge` | 3 | `identify_steps`, `classify_step`, `estimate_eval_turns` |
| `agent_loop` | 3 | `codification_scan`, `design_signatures`, `assemble_ir` |
| `hitl_gate` | 0 | (none) |

### HITL gates

None. The rote-compile skill has no explicit human-approval signal. The
compile report (Phase 7) is a terminal output, not a gate. The pipeline
writes it to disk and completes.

### Parallelism structure

The compiled pipeline has four concurrent waves within the main compilation
path:

1. **Wave 1:** `read_skill_bundle` ‖ `detect_entry_point`
2. **Wave 2:** `identify_steps`
3. **Wave 3:** `classify_step` (fan_out per identified step)
4. **Wave 4:** `codification_scan` ‖ `design_signatures` ‖ `estimate_eval_turns` (fan_out)
5. **Wave 5:** `assemble_ir`
6. **Wave 6:** `validate_ir` → `write_eval_sidecar` (fan_out results collected)
7. **Wave 7:** `invoke_adapter` ‖ `write_compile_report`

The biggest wall-clock win is Wave 4: codification scanning, signature design,
and eval turn estimation all run concurrently once classification is done.

---

## Phase 2: Node classification table

| Step | Kind | Mandatory | Justification |
|------|------|-----------|---------------|
| `read_skill_bundle` | `external_call` | Yes | Deterministic filesystem reads that need retry semantics. Every other node depends on this output. |
| `detect_entry_point` | `pure_function` | Yes | Fixed mapping from CLI flags to phases and mode. No LLM or external service involved. |
| `identify_steps` | `llm_judge` | Yes | Reading prose and naming distinct steps is fuzzy but produces bounded output: a list with a fixed StepDescription schema. |
| `classify_step` | `llm_judge` | Yes | Applying the 5-kind rubric to a described step is bounded classification. Output is an enum kind plus text justification. Fan_out per step. |
| `codification_scan` | `agent_loop` | N/A | Exploratory search for crystallizable patterns. The agent decides which files to re-read and in what order. Genuinely varies per skill. |
| `design_signatures` | `agent_loop` | N/A | Designing typed schemas requires reading rubric sections and making design decisions. Exploratory per skill. |
| `estimate_eval_turns` | `llm_judge` | No | Given step and kind, estimating turn bounds is bounded fuzzy judgment. Output has a fixed schema (low/high/iterations). Fan_out per classified step. |
| `assemble_ir` | `agent_loop` | N/A | Synthesizing all prior work into pipeline.yaml requires many judgment calls about data flow, timeouts, and constants. Varies per skill. |
| `write_eval_sidecar` | `pure_function` | Yes | Renders eval.yaml from structured inputs using a fixed schema. No LLM needed. |
| `validate_ir` | `pure_function` | Yes | Deterministic call to `rote.ir.load_pipeline()`. Same input always produces the same validation result. |
| `invoke_adapter` | `external_call` | No | Runs `rote emit` as a subprocess. Skipped in report_only mode. |
| `write_compile_report` | `pure_function` | Yes | Renders compile-report.md from structured inputs using a fixed Markdown template. |

Note: `mandatory: true` cannot be set on `agent_loop` nodes per the IR schema.
The agent loops are enforced as non-optional by the DAG structure (downstream
nodes need their output), not by the mandatory flag.

---

## Phase 3: Crystallization log

### Extraction 1: `read_skill_bundle`

**Source:** `SKILL.md` §Phase 1: Intake
**Before (prose):**
> "Read the target directory exhaustively: SKILL.md (parse the frontmatter)... references/ (read every sub-file)."

**After (code):** `extracted/file_reader.py:read_skill_bundle`

The extraction is a complete working implementation (not a stub), since the
file I/O contract is fully specified in prose and requires no external API
knowledge. Constants lifted:
- `SKILL_MD_FILENAME = "SKILL.md"`
- `REFERENCES_SUBDIR = "references"`

---

### Extraction 2: `detect_entry_point`

**Source:** `SKILL.md` §Phase Routing
**Before (prose):**
> "Full compilation: Phase 1 → 7 | Report only: Phase 1 → 5, skip 6 | Re-emit for different runtime: Phase 6 | Update compilation after skill changed: Phase 1 → 7, diff"

**After (code):** `extracted/entry_point.py:detect_entry_point`

The Phase Routing table maps directly to an enum + list of phase names.
Constants lifted:
- `PHASE_NAMES = ["intake", "node_classification", ...]` (all 7 phase names)
- `REPORT_ONLY_PHASES` (the subset run for report-only mode)

The "re_emit" mode (Phase 6 only) is not exposed as a CLI flag in this
compilation. It maps to running `rote emit` directly. This is an open question
(see below).

---

### Extraction 3: `validate_ir`

**Source:** `SKILL.md` §Phase 6: Adapter Invocation
**Before (prose):**
> "Checklist before declaring Phase 5 done: Every pure_function/external_call node has an impl:... Every llm_judge node has a signature:..."

**After (code):** `extracted/ir_validator.py:validate_ir`

The Phase 6 checklist is enforced programmatically by `rote.ir.load_pipeline()`,
which runs all checklist items as Pydantic validators. This is the highest-value
extraction: a MANDATORY checklist enforced only by prose becomes impossible to
skip once it's in code.

---

### Extraction 4: `invoke_adapter`

**Source:** `SKILL.md` §Phase 6: Adapter Invocation
**Before (prose):**
> "The `rote emit <pipeline.yaml> --runtime temporal --out <dir>` CLI command (or its programmatic equivalent) takes it from there."

**After (code):** `extracted/adapter_invoker.py:invoke_adapter`

The subprocess call has fixed semantics. Given a valid IR and a runtime name,
it always produces the same set of emitted files. The report_only short-circuit
is also deterministic (returns empty list immediately).

---

### Extraction 5: `write_eval_sidecar`

**Source:** `SKILL.md` §Phase 5: IR Assembly + `references/eval-estimates.md`
**Before (prose):**
> "Produce eval.yaml next to pipeline.yaml: per-step agent-turn estimates..."

**After (code):** `extracted/eval_sidecar_writer.py:write_eval_sidecar`

The eval.yaml schema is fully specified in eval-estimates.md. The renderer
takes structured inputs (lists of estimates + step metadata) and produces
the fixed YAML format.

---

### Extraction 6: `write_compile_report`

**Source:** `SKILL.md` §Phase 7: Compilation Report
**Before (prose):**
> "Produce a human-readable Markdown report in compile-report.md covering: Summary metrics, Crystallization log, Open questions, Suggested next steps."

**After (code):** `extracted/compile_report_renderer.py:write_compile_report`

The four-section structure of the report is fixed. Given typed inputs
(classified_steps, codification_plan, signature_designs, validation_result),
the render is purely mechanical.

---

## Phase 4: Signature designs

### `identify_steps` (llm_judge)

- **File:** `signatures/identify_steps.py:IdentifySteps`
- **Evals:** `evals/identify_steps.jsonl` (3 seed examples)
- **Input schema:** `{skill_md: str, reference_files: dict[str, str]}`
- **Output schema:** `{steps: list[StepDescription]}` where StepDescription has `{step_id, name, description, source_section, phase?}`
- **Pre-filter:** Short-circuits if SKILL.md body is fewer than 5 lines (validates the input)
- **Source rubric:** SKILL.md §Phase 2 (step granularity rules)

---

### `classify_step` (llm_judge)

- **File:** `signatures/classify_step.py:ClassifyStep`
- **Evals:** `evals/classify_step.jsonl` (6 seed examples, one per common pattern)
- **Input schema:** `{step: StepDescription, skill_md: str, reference_files: dict}`
- **Output schema:** `{step: StepDescription, kind: NodeKind, justification: str, suggest_mandatory: bool}`
- **Pre-filter:** Short-circuits on naming-convention hints (`write_*` → `pure_function`, `read_*`/`validate_*` → `external_call`) before calling the LLM
- **Source rubric:** `references/node-kinds.md` (all five kind descriptions + decision rules)

---

### `estimate_eval_turns` (llm_judge)

- **File:** `signatures/estimate_eval_turns.py:EstimateEvalTurns`
- **Evals:** `evals/estimate_eval_turns.jsonl` (4 seed examples)
- **Input schema:** `{step: ClassifyStepOutput, skill_md: str}`
- **Output schema:** `{turns_low, turns_high, has_iterations, iterations_low?, iterations_high?, notes?}`
- **Pre-filter:** Short-circuits on `hitl_gate` (0/0 turns) and deterministic kinds (`pure_function`/`external_call`, 1-3 turns) before calling the LLM
- **Source rubric:** `references/eval-estimates.md` (calibration anchors + schema)

---

## Phase 6: Adapter invocation readiness checklist

| Requirement | Status |
|-------------|--------|
| Every `pure_function`/`external_call` has `impl:` | ✅ All 6 nodes have impl: pointing at extracted/ stubs |
| Every `llm_judge` has `signature:` and `signature_spec:` | ✅ All 3 judges have both |
| Every `agent_loop` has `tools:` | ✅ All 3 loops have tools: + tool_servers: |
| No `hitl_gate` nodes | ✅ None in this pipeline |
| All `inputs:` references use the 4-form grammar | ✅ Verified. No arithmetic or deep paths. |
| All node references in `edges`, `entry_nodes`, `exit_nodes` are valid | ✅ All 12 node IDs match |
| No `on:` YAML keys | ✅ Uses `retry_on:` throughout |
| `mandatory: true` absent on `agent_loop` nodes | ✅ Only `pure_function`/`external_call`/`llm_judge` carry mandatory: true |
| MCP bindings on MCP tool calls | ✅ All three agent loops declare `tool_servers:` with `filesystem` server |

---

## Open questions

### 1. `identify_steps` granularity calibration

This is the most important open question. `identify_steps` produces the list
that `classify_step` fans over. If it over-merges steps (e.g., treating the
whole "Phase 3: Codification Scan" as one step when it has two sub-steps:
"scan for patterns" and "write extracted stubs"), downstream classification
will be coarse and the compiled pipeline will have under-specified nodes.

**Decision made:** The prompt instructs the model to err toward more granular.
**Verify:** Run the pipeline on BDR and check whether `identify_steps` produces
a step list that matches the hand-crafted BDR pipeline's node count (±2).

---

### 2. `assemble_ir` agent loop max_iterations

The current IR sets `max_iterations: 5` for `assemble_ir`. For a complex skill
with 15+ nodes, the agent may need more iterations to produce a valid pipeline.yaml
(each failed validate_ir call is one iteration). The value was chosen conservatively.

**Decision made:** 5 iterations. If the first production run hits the cap,
increase to 8.

---

### 3. `detect_entry_point`: the "re_emit" mode is not modeled

The SKILL's Phase Routing defines four modes: full, report_only, update, and
re_emit. This compilation models only three (full, report_only, update). The
"re_emit" mode (which starts at Phase 6 with an existing pipeline.yaml) is
not a natural fit for this pipeline's DAG (it would need to skip phases 1-5).

**Decision made:** Users who want re_emit behavior should call `rote emit`
directly (which is what the skill says anyway). The compiled pipeline handles
three modes.

---

### 4. `estimate_eval_turns.inputs.step` references `classify_step.output`

The `estimate_eval_turns` node is fan_out: true over `classify_step.output`.
Since `classify_step` is also fan_out: true, `classify_step.output` is the
collected list of all `ClassifyStepOutput` objects (one per identified step).
The estimate_eval_turns node fans over that collected list.

This double fan-out pattern (fan → collect → fan again) is supported by the
rote adapters but should be verified against the DBOS and Temporal adapters'
emission logic specifically.

**Decision made:** Modeled as double fan_out with intermediate collection.
**Verify:** Run `rote emit pipeline.yaml --runtime temporal` and check
the emitted `activities.py` for `estimate_eval_turns`.

---

## Suggested next steps

1. **Dogfood on the BDR skill.** Run this compiled pipeline on
   `examples/bdr-outreach/skill/` and compare the output `pipeline.yaml`
   against the committed snapshot in `examples/bdr-outreach/expected/`. This
   is the fastest calibration check for `identify_steps` and `classify_step`.

2. **Run `rote emit pipeline.yaml --runtime python`** as the smoke test.
   The Python adapter produces the most legible output (one function per node),
   so a human reviewer can quickly verify that the DAG structure matches the
   intended data flow.

3. **Expand `evals/classify_step.jsonl`** after the first real compilation run.
   The six seed examples cover the five kinds plus the "agent_loop vs.
   llm_judge" ambiguity. After classifying 2-3 real skills, add the cases where
   the model's first classification disagreed with the reviewer. Those are the
   highest-value examples.

4. **First node to recompile after production data:** `estimate_eval_turns`.
   The calibration anchors in the pre-filter (1-3 turns for pure_function,
   3-15 for agent_loop) are derived from measured BDR data. After running
   `rote eval --run` on the first real compilation, compare measured vs.
   estimated turns and update the constants in `signatures/estimate_eval_turns.py`
   if they're systematically off.

5. **Add a `load_existing_ir` node** if the `--update` mode sees heavy use.
   The current compilation models update as "re-run everything with update=true
   passed to detect_entry_point". A production-grade update mode would need
   a separate `load_existing_ir` (external_call) node at entry and a
   `diff_provenance` (pure_function) node to determine which nodes to skip.
