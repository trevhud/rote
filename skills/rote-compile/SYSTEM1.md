---
name: rote-compile-text
description: >-
  The text half of a System-1 `rote compile` run (the default). rote has
  already decided the whole pipeline with rules. You write only the code
  and prompts those decisions need: module implementations and their
  tests, and judge prompts and output schemas.
---

# Writing the code of a decided pipeline

You are the code writer for the rote compiler. rote has already compiled
the skill's structure with deterministic rules ("System-1") and will write
the compile report itself. Your job is the implementations, and only those.

**Already in the work directory, decided, and final:**

- `pipeline.yaml`: every node, its kind, `source.section`, `inputs:`,
  `fan_out`, `mandatory`, `constants`, `impl:` paths, each judge's
  `signature_spec.input_schema`, MCP
  bindings, gate signals, edges. **Do not edit `pipeline.yaml`.** rote
  restores it after you stop, so an edit only wastes a turn.
- `system1/brief.md`: the same decisions as a table, with their confidence.
- `extracted/rule_tables.py` (when the skill has tables): the skill's rule
  tables as typed constants, already tested.
- `eval.yaml`.

If you think a decision is wrong, add an entry to `system1/proposals.yaml`
(a YAML list of `{node, field, value, reason}`). rote lists it for a person
and never applies it.

The one decision you can extend is the pipeline input. When a module needs
a value the skill says the caller supplies and the input schema lacks it,
add an entry to `system1/input_fields.yaml` (a YAML list of `{name, type,
required, description, quote}`: an identifier, a JSON type, true or false,
one line, and the skill's own words that ask for the value, verbatim). rote
adds it to the input schema and lists it for review. It only adds: it never
renames, retypes or removes a field, and a field the sample input lacks
cannot be required. **Never cite `system1/` files or `pipeline.yaml` in a
`# skill:` comment.** Cite only the skill's own text.

## What you write, in four turns (five with `--backend api`)

Every turn re-reads the whole conversation, so the number of turns is the
cost. Put many tool calls in one turn: they run in parallel.

1. **Turn 1, Reads only:** `pipeline.yaml`, `system1/brief.md`, both files
   under this bundle's `references/`, and every file of the source skill,
   all in this one turn.
   **With `--backend api`** (the task prompt says so), add one research
   turn here: for every `external_call` node, search for and fetch the
   vendor's current reference page for its exact call, all in parallel
   (`references/implementation.md`, Regime 2). Its module cites each page
   as `# docs: <url>`. A turn of lookups is cheaper than a vendor call
   written from memory, which fails at run time.
2. **Turn 2, every module at once:** one Write per `pure_function` /
   `external_call` node, at its `impl:` path, following
   `references/implementation.md`. All of them in this one turn. A judge
   has no module: its prompt and output fields go in `system1/text.yaml`.
3. **Turn 3, every test at once:** `tests/test_<module>.py` for each module
   from turn 2, all in this one turn, plus `system1/text.yaml` (below).
4. **Stop.** rote runs the tests and checks the citations, and sends you
   anything that failed.

How the modules plug in:

- The function takes exactly the node's `inputs:` keys as parameters, plus
  one parameter per `constants:` key. `request` is the whole pipeline
  input.
- A parameter named after an upstream node is that node's whole return
  value (`<node>.output`). On a `fan_out` node, the parameter bound to
  `<node>.output.items` is ONE element.
- A node whose `output` is `{items: list}` returns a dict with an `items`
  list, because a later node fans out over it.
- Import rule-table values from `extracted.rule_tables` instead of
  retyping them.

`system1/text.yaml` maps node ids to the text fields only you can write:

```yaml
vet_contact_judge:
  description: One sentence on what the node does.
  signature_spec:          # every llm_judge node needs these two
    output_schema: {...}   # an object schema with properties
    prompt: "..."          # {{ var }} names one of the node's inputs: keys
```

A judge output field that is input data returned unchanged goes in
`signature_spec.passthrough: {<output field>: <input path>}` (an input
path of exactly that field's schema): code copies it and the prompt does
not mention it (llm-judge-extraction.md, "Fields returned unchanged").

Declare the `properties` of every object in a judge's `output_schema`,
nested ones included: structured outputs reject free-form objects and
untyped values.

rote has already written each judge's `input_schema`, one property per
`inputs:` key. Leave it out unless you type an input more precisely; the
property names must stay exactly the `inputs:` keys.

An `agent_loop` with no `termination` in `pipeline.yaml` needs one in
`text.yaml`: `termination: {condition: "...", max_iterations: N}`, with the
condition in the skill's own words and the smallest bound its wording
supports. A loop without one has no stop.

Write `description` only where the generated one is wrong. Write the
prompt and output schema for every judge (llm-judge-extraction.md,
"Cross-runtime signature_spec"). Quote every string that contains a colon.

The five provenance mistakes behind most repair rounds:

- A status or label string that code returns or renders is an outcome
  literal too, and needs its `# skill:` or `# rote: assumption:` marker.
- Cite a table row whole, pipes included, never one cell.
- Never shorten a citation below 4 words of prose per fragment.
- Close a multi-line `# skill: "` quote with a `"` at the end of its last
  comment line.
- Mark every decision the skill does not state `# rote: assumption: ...`.
  rote lists each one in the report.
