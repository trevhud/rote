# System-1 compile: rules decide, the LLM writes code

Status: the default `rote compile` path. `--no-system1` runs the
full-rubric agent instead, and `--update` always does, because System-1
does not support incremental re-compilation yet. This document was written
while it was an experimental `--system1` flag, so the measurements below
call it that. Built on main after #144 (`--effort low`) and #146 (`rote bench`).

The principle, in the user's words: *"classification is something we can do
with a system one model. Also, any decision. We should only be using LLMs
when we actually need text output."* The priorities, in order: a converted
skill must behave like the original, then speed, then cost.

## Measured (rote bench, 5 skills, 1 repeat, same day)

| skill | main + #144 | `--system1`, text on Sonnet 4.6 (v1) | `--system1`, text on Sonnet 5.5 (v2) |
|---|---|---|---|
| expense-audit | $1.00, 362 s, 83 tests | $1.21, 471 s, 69 tests | **$0.51, 175 s**, 39 tests |
| ops-report | $1.01, 404 s, 51 tests | $0.77, 263 s, 46 tests | **$0.36, 107 s**, 18 tests |
| invoice-push | $1.42, 527 s, 41 tests | $1.92, 564 s, 75 tests | **$0.55, 187 s**, 46 tests |
| deal-monitor | $1.68, 600 s, 70 tests | $1.24, 390 s, 51 tests | **$0.46, 156 s**, 29 tests |
| bdr-outreach | $1.91, 850 s, 53 tests | $0.97, 336 s, 35 tests | **$0.38, 101 s**, 23 tests |
| **mean $/compile, wall** | **$1.40, 548 s** | $1.22, 405 s | **$0.45 (-68%), 145 s (-74%)** |

Every compile in all three columns verified, with tests passing, 0 citation
errors, and dbos and cloudflare emits ok. v2 also has the deterministic report,
`text.yaml` and the four-turn rubric (below). Comparing expense-audit on v2
with Sonnet 4.6 ($0.81, 314 s, 73 tests) separates the two effects: the
rubric and report changes cut cost by a third, and the move to Sonnet 5.5
($2 / $10 against $3 / $15 per MTok) cuts it by another third. Sonnet 5.5
writes about half as many tests for the same verified result. Structural
scores were identical on both models, which is expected when the structure is
decided before the model runs.

Structure, means over the 5 skills:

| metric | main + #144 | v1 | v2 |
|---|---|---|---|
| node recall / precision | 0.81 / 0.79 | 0.58 / 0.55 | 0.56 / 0.53 |
| kind accuracy (name-matched) | 0.68 | 0.51 | 0.45 |
| kind mix (matching-free) | 0.81 | 0.80 | **0.82** |
| edge F1 / wave order | 0.73 / 0.92 | 0.75 / 0.92 | 0.57 / 0.94 |
| fan_out precision / recall | 0.25 / 0.12 | 1.00 / 0.38 | **0.75 / 0.38** |
| mandatory precision / recall | 0.47 / **0.86** | 0.25 / 0.04 | 0.32 / 0.43 |
| gate signal match / payload Jaccard | **0.4 / 0.29** | 0 / 0 | 0 / 0.17 |
| input field recall / precision | **0.38 / 0.57** | 0 / - | 0.09 / 0.25 |

Read this table with the benchmark's matcher in mind. It pairs nodes by id
and description tokens, so the curated names (`exclusion_check_dnc`) favor
an agent that invents similar ones. A System-1 node that does exactly the
right thing under a heading-derived name counts as both missing and extra,
and its mandatory and fan_out flags are scored against nothing.
`kind mix`, which is free of matching, is where System-1 ties or leads.
The edge F1 drop from v1 to v2 is a matching effect too: shorter ids and
the per-item loop moved which nodes pair up.

## What decides what

| decision | rule | confidence |
|---|---|---|
| which steps exist | leaf procedural sections. Meta sections (trigger phrases, routing, tool tables, limits) are skipped. A heading that names no action under a step ("Tab 1: Invoice Detail") is part of that step. An enumerated list splits a step only when most items carry a step signal, so a checklist ("Browser-automation plugin", "User is logged in") stays one step | n/a |
| per-item loops | a container heading whose sub-steps run "for each" row, item or record ("Process Each Invoice Row") becomes an `agent_loop` whose `loop_body` is those sub-steps. Run once, they would process one row | 0.7 |
| nodes per step, and kinds | node-kinds.md in order: a human gate wins. An exploratory loop wins over its tools. A tool call is a bare `external_call`, and the work on its result is its own judge or pure_function. "Group ... by X, for each X" is two nodes. A write is its own `external_call`. A UI step (click, navigate, select) gets a pure_function only when it also describes a computation (convert, total, count, format, parse ...). No signal is one `pure_function` | 0.9 one kind alone, 0.8 dominant, 0.6 contested, 0.3 no signal |
| data flow | explicit step references ("flags from steps 2 and 3"). "Steps N to M do not depend on each other" makes them siblings. Enumerated sub-steps are siblings. Otherwise the step before. Every node also reads `request: pipeline.input` | 0.9 explicit, 0.8 stated parallel, 0.7 siblings, 0.5 default |
| fan_out | per-item wording ("for each X", "each X", "per X", "one at a time") on a judge or a "for each" work node, over the node before it, which then outputs `{items: list}`. "Per" followed by a unit of measure ("10 per call", "$75.00 per night") does not count. A pure_function never fans out on wording alone: looping in code gives the same result, while one judge call per item is a different call from one over the list | 0.7 |
| mandatory | an explicit marker (MANDATORY, "required before"), or else every step with no conditional wording ("optionally", "only if", "if needed", "skip", "may"). Gates are mandatory implicitly and not flagged | 0.95 / 0.6 |
| gates | signal `<gate>_approved`, or `_provided` when the person supplies data. Payload keys are the items the step asks for in bold, else the decisions it offers ("approve or reject" gives `approved`, `rejected`), else `approved` + `notes`. Timeout is the LONGEST wait stated ("usually within a day ... up to three days" parks 3d) | 0.8 / 0.5 |
| pipeline input fields | placeholders in the skill's prose ("for [drug]", "for <month>"), and the items an opening step asks the person to confirm. Placeholders inside code are skipped (`[start date]` in a formula is computed, not supplied). None is required | n/a |
| constants | comparators, "more than N", "N or more", batch sizes, windows, limits, page sizes, file/channel/folder ids. One parameter per distinct value. Ids and windows go to the fetch, thresholds to the work | 0.8 |
| MCP bindings | "(Gmail MCP: search_threads)" or a tool from the skill's tool tables. A server named without a tool gets no binding | 0.9 / 0.3 |
| agent-loop tools | tools named in the step, else the search tools of servers it names, else a placeholder | 0.9 / 0.6 / 0.2 |
| rule tables as code | every markdown table with numeric columns becomes `{key: Decimal}` in `extracted/rule_tables.py`, with each row cited verbatim and a generated test. Sample tables (rows numbered 1, 2, 3, or cells elided with "...") are skipped | exact |
| node ids | at most four content words of the heading, with a role prefix or suffix. A sub-step takes its heading's first word as a prefix | n/a |

## What the paid model still does, and how little

The text agent gets `skills/rote-compile/SYSTEM1.md` (about 90 lines) plus
`implementation.md` and `llm-judge-extraction.md`: about 890 lines, against
about 2,400 for the full rubric. Its job is four turns:

1. Read everything, in one turn.
2. Write every module, in one turn.
3. Write every test and `system1/text.yaml` (judge `signature_spec`s and
   any better descriptions), in one turn.
4. Stop.

Three changes moved paid work into code. Each came from reading a v1
transcript:

- **The report is generated.** rote renders `compile-report.md` from data
  with the shared renderer both modes use (`rote.compiler.report`, #152).
  `--system1` passes `system1.report_extras`: a preamble, one Open
  questions entry per low-confidence decision, per proposal and per
  reverted edit, and its next steps. The agent spends no turn on it.
- **Text goes in one file.** In v1, one compile spent 9 of 44 turns on
  separate Edits to `pipeline.yaml` descriptions. The agent now writes
  `system1/text.yaml` once, and rote merges only text fields from it.
- **Writes are batched.** v1 wrote one module per turn, so each turn
  re-read a longer conversation. v2 averages 33 turns against v1's 47.

## Decisions System-1 got wrong (v2), by behavioral impact

1. **Mandatory.** Recall rose from 0.04 to 0.43 under the "unless
   conditional" rule, and precision is 0.32. In the fidelity direction, an
   extra flag costs only precision, while a missing one lets a step be
   skipped. Most of the 11 "missed" flags sit on nodes the matcher did not
   pair, not on unflagged steps.
2. **fan_out.** Misses: `personalize_email` (BDR, "Email Templates" names
   no judging verb), `classify_invoice_row` (now runs inside the per-item
   loop, so per-row is still right), and `score_new_opportunities`, where
   the node is fanned out but the matcher paired it differently.
3. **Gate contracts.** Payload keys come from the skill's words (the
   ops-report gate's three items match the curated keys in meaning). The
   curated names (`approved_ids`, `rejected_ids`) are not in the skill's text.
4. **Input fields.** Only what the skill names itself. Curated fields like
   `submissions` and `report_date` are not in the text, so no rule can
   produce them without inventing them.
5. **Segmentation.** Invoice-push has 22 nodes against 13: a UI step such as
   "Clear All Existing Filter Pills" becomes pure + write. BDR's lead
   generation loop has no `enrich` / `vet` sub-nodes.

## Held-out set: the rules do not generalize yet

Three skills under `examples/holdout/` were written by another agent with
gold labels and kept blind: I never opened their skill or expected files.
I looked only at the scores and the aggregate error counts the bench
prints. Both modes ran with one repeat each. The rules were frozen at
8ab7ad4 (system1, Sonnet 5.5 for the text half), and the default path ran
on main at ffd8a73.

| skill | default: $, wall, node recall, kind mix, gates | `--system1`: $, wall, node recall, kind mix, gates |
|---|---|---|
| customer-onboarding | $0.97, 333 s, 0.80, 0.91, 1/1 | $0.20, 50 s, 0.60, 0.60, 1/1 |
| incident-triage | $1.77, 680 s, 0.79, 1.00, 1/1 | $0.50, 164 s, 0.43, 0.71, **0/1** |
| saas-spend-audit | $1.59, 436 s, 0.82, 0.92, 1/1 | $0.42, 145 s, 0.18, 0.64, **0/1** (the gate became an external_call) |
| **mean** | **$1.44, 483 s** | **$0.37, 120 s** |

Every compile in both modes verified, with tests passing and both emits ok.

| metric (mean) | default, dev | default, held-out | system1, dev | system1, held-out |
|---|---|---|---|---|
| kind mix | 0.81 | 0.94 | 0.84 (8ab7ad4, at $0) | **0.65** |
| node recall | 0.81 | 0.80 | 0.56 | 0.40 |
| edge F1 | 0.73 | 0.88 | 0.57 | 0.52 |
| fan_out P / R | 0.25 / 0.12 | 0.75 / 0.75 | 1.00 / 0.38 | 0.00 / 0.00 |
| mandatory P / R | 0.47 / 0.86 | 0.71 / 0.71 | 0.32 / 0.43 | 0.48 / 0.36 |
| gates found | 4/4 | 3/3 | 4/4 | **1/3** |
| input field recall | 0.38 | 1.00 | 0.09 | 0.00 |

The default path holds or improves on new skills. System-1 drops on every
structural measure, and it **misses two of three human gates**. That is a
fidelity failure, not a naming one: a missed gate ships a pipeline that
no longer waits for the person the skill asks. The error counts point at
vocabulary. External calls and pure steps went missing, and a gate was
read as a call. The rules cover the wording of the five dev skills and
not much past it.

These failures were not tuned on. The rule changes that came after the run
(below) came from the #150 review's wording, not from the held-out skills.
Re-scoring them on the held-out set would still not be blind, because the
error types above are now known.

## Rules at risk of not generalizing

Listed here whatever the held-out scores show, as the #150 review asked:

- `parse.py` SIGNALS. Kinds come from a fixed vocabulary of verbs. Until
  bb1f261 the fetch signal keyed on a vendor list (slack, gmail, hubspot,
  zoominfo ...) and the gate signal on role names (duty manager,
  controller, bdr). After the held-out run, both were replaced by general
  patterns: a retrieval verb with a source, and anyone asked to confirm,
  approve, review, provide or sign off. Those are tested on tiny skills
  outside the examples. On the dev set this costs kind mix 0.839 to
  0.817, because deal-monitor's "Read the entire thread" no longer reads
  as a call.
- `_PER_ITEM` (per-item containers) said "each visible row|invoice".
  It now matches any "for each" / "each X".
- The constants window pattern had a Gmail-only `newer_than:`, now removed.
- `decide.py` segmentation order and its confidences (0.9 / 0.8 / 0.6 /
  0.3) were set by reading the dev skills, not fitted.
- `_UI_ACTION` and `_TRANSFORM` (UI steps stay whole unless they compute)
  are vocabularies too.
- The parallel-steps and step-reference patterns need a skill to number its
  steps.

What generalizes by construction: explicit MANDATORY markers, "for each",
"do not depend on each other", stated durations, tool tables, markdown
rule tables, and the conservative defaults. A missing signal falls to a
listed low-confidence decision, never to a guess.

## `--decider clef`

Clef (`@cf/cloudflare/clef` on Workers AI, a decision model with no generated
text) decides each step's kind and whether a person must act. It uses the
`models` lane's pre-registered phrasings, ensemble and thresholds. Clef wins
when its ensemble kind confidence is at least 0.85 or its gate probability is
at least 0.50. Otherwise the rules decide, and a disagreement is listed in
Open questions. Mandatory and fan_out stay on rules. A missing credential
fails the compile (`CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_API_TOKEN`).
`CLOUDFLARE_AI_GATEWAY_ID` routes the calls through an AI Gateway, and
`CLOUDFLARE_AI_GATEWAY_METADATA` (a JSON object) tags each logged call, which
is how rote cloud bills one compile's Clef calls to its tenant. A
spot check of three dev steps against the lane's cached answers agreed
within 0.01 on kind and on which side of the gate threshold each fell.

Skeletons scored with the structural matcher (#151). Structure is fixed
before the agent runs, so these are the compiled structure. The held-out
set is not blind any more:

| split | decider | node recall | kind | kind mix | gates found | false gates |
|---|---|---|---|---|---|---|
| dev (5) | rules | 0.76 | 0.64 | 0.80 | 4/4 | 1 |
| dev (5) | clef | 0.76 | 0.64 | 0.80 | 4/4 | 2 |
| held-out (3) | rules | 0.74 | 0.49 | 0.67 | 2/3 | 0 |
| held-out (3) | clef | 0.74 | **0.64** | **0.74** | **3/3** | 0 |

Clef's extra dev gate is invoice-push's "Prerequisites Check" (a person
must be logged in, p = 0.54). Clef cost $0.036 across the 8 skills, about
$0.0045 a compile.

Behavioral fidelity (`rote bench --fidelity`, Sonnet 5.5 text, one compile
each):

| skill | rules | clef | default path (from the fidelity table) |
|---|---|---|---|
| ops-report | 0.83 | 0.81 | 0.75 |
| expense-audit | 0.11 | 0.11 | 0.17 |

Expense-audit fails in both modes on the input contract: the code reads
`request["lines"]`, and the skill's input arrives as `period` +
`submissions`. That is the `grounding` lane's fix (sample input as the
contract), not a decider difference.

## `--decider jev`

Jev (`typesafe/jev-1.14.0`, Typesafe's decision model on Workers AI) takes
the same state and typed questions as Clef, so it runs through the same
client (`src/rote/compiler/system1/decision_model.py`) with its own route
and thresholds. It is served on the unified `/ai/run` route, which takes
the model id in the body and the request under `input`, and wraps the
answer once more (`result.result`).

The phrasings and the ensemble are Clef's, unchanged. Only the two
thresholds were fitted, on the five dev skills only, by a rule written
down before the fit (`research/system1-routing/score_deciders.py fit`):

- Kind: the lowest threshold at which Jev's kind choice is one of the
  step's expected kinds on at least 95% of the dev steps at or above it,
  with at least 5 such steps. That is 0.85 (21 of 22 steps right).
- Gate: the threshold with the best gate F1 on dev steps, the higher one
  on a tie. That is 0.60 (4 of 4 gates, 1 false).

The same rule applied to Clef gives 0.90 and 0.60, not its frozen 0.85
and 0.50. Clef keeps its frozen values, because changing them now would
be fitting on the held-out set.

Then one measurement on the held-out set, with both frozen
(`score_deciders.py score`, skeletons scored with `score_pipeline`):

| split | decider | kind | kind mix | gates found | false gates | mandatory P / R | $ / compile |
|---|---|---|---|---|---|---|---|
| dev (5) | rules | 0.64 | 0.78 | 4/4 | 1 | 0.40 / 0.49 | 0 |
| dev (5) | clef | 0.64 | 0.78 | 4/4 | 2 | 0.39 / 0.45 | 0.0049 |
| dev (5) | jev | 0.62 | 0.78 | 4/4 | 2 | 0.39 / 0.45 | 0.0011 |
| held-out (3) | rules | 0.49 | 0.71 | 2/3 | 0 | 0.94 / 0.68 | 0 |
| held-out (3) | clef | 0.66 | **0.74** | **3/3** | 0 | 1.00 / 0.68 | 0.0038 |
| held-out (3) | jev | **0.67** | 0.72 | **3/3** | 1 | 1.00 / 0.65 | 0.0008 |

On the held-out skills both models beat the rules on kind and find the gate
the rules miss. Jev matches Clef on kind at a fifth of the price and adds
one false gate. On dev neither changes the kinds the rules chose, and each
adds one false gate (invoice-push's "Prerequisites Check"). Mandatory is a
rule decision for every decider; it moves only because a model's gate
removes a node the rules marked mandatory.

These rows differ slightly from the Clef table above (0.66 against 0.64
held-out kind) because the parser and the argument binding changed since
(#155 to #160).

The built-in default stays `rules`: no credential, and nothing leaves the
machine. The held-out numbers argue for a model decider as the default, so
that is a decision for the user, not this change. The decider is a layered
setting (`decider:` in `rote.yaml` or the user config, `ROTE_DECIDER`,
`--decider`), and `rote init` asks for it.

## The Haiku tie-break (`--tiebreak haiku`, the default)

After the decider, a step whose kind decision is still below
`REVIEW_BELOW` (0.7) is unsure: the rules had no clear signal, or a
decision model disagreed below its threshold. The unsure steps of one
compile go to `claude-haiku-5-5` in one schema-locked call
(`src/rote/compiler/system1/tiebreak.py`), with node-kinds.md, the step
list, and each unsure step's text, rule guess and model guess. Haiku's node
list replaces the conservative default for those steps and nothing else.
Its decisions are settled (confidence 0.7, the rule names the model and its
reason) and are recorded under `tiebreak` in decisions.json with their usage
and cost.

The call bills the way the compile does: `claude -p` on the subscription
under the claude driver, the api driver's own client (gateway base URL,
headers and model-id prefix) under `--agent api`, otherwise an Anthropic
key, then the CLI. With none of them the compile fails and names
`--tiebreak none`. It goes through `structured_call` in
`rote.inference._runtime_helper`, the code emitted judges run: a forced
tool call, no temperature, all four usage buckets.

This replaces the earlier offline replay (`route_haiku.py`), which scored
an older rule set (0.456 to 0.531 kind). Nothing was fitted for it: the
threshold is the existing `REVIEW_BELOW`. Measured on skeletons
(`score_deciders.py score --tiebreak 3`, three trials each, means):

| split | decider | kind | + tie-break | kind mix | + tie-break | gates, false gates (+ tie-break) | unsure steps | tie-break $ / compile |
|---|---|---|---|---|---|---|---|---|
| dev (5) | rules | 0.64 | 0.63 | 0.78 | 0.81 | 4/4, 1 (same) | 14 | 0.001 |
| dev (5) | clef | 0.64 | 0.64 | 0.78 | 0.80 | 4/4, 2 (same) | 16 | 0.001 |
| dev (5) | jev | 0.62 | 0.67 | 0.78 | 0.77 | 4/4, 2 (same) | 20 | 0.001 |
| held-out (3) | rules | 0.49 | **0.68** | 0.71 | 0.77 | 2/3, 0 (one trial 1) | 8 | 0.001 |
| held-out (3) | clef | 0.66 | **0.74** | 0.74 | 0.76 | 3/3, 0 (same) | 6 | 0.001 |
| held-out (3) | jev | 0.67 | **0.72** | 0.72 | 0.74 | 3/3, 1 (same) | 5 | 0.001 |

On the held-out skills the tie-break adds 0.19 kind accuracy to the rules,
0.08 to Clef and 0.05 to Jev. On dev it is flat for rules and Clef (the
rules were written on those skills, so their unsure steps are the hard
ones) and adds 0.05 to Jev. Clef plus the tie-break is the best held-out
row. It never finds the gate the rules miss: gates are a decider's job.
One call adds about 6 s to a compile.

## Non-generative models, measured

- **Local embeddings, nearest neighbor**, leave one skill out (52 labeled
  steps). `BAAI/bge-small-en-v1.5` (fastembed) reaches 0.40 / 0.56 section
  kind recall / precision, and `minishlab/potion-base-8M` (model2vec) 0.37 /
  0.37, against the rules' 0.75 / 0.72. Overriding a rule's kind with the
  neighbor's never helps at any similarity cut. Neighbors learn each skill's
  domain, not its step kinds.
- **Cloudflare Clef, Clef-flash, Clef-omni and Jev** (decision models on
  Workers AI, measured by the `models` lane on these units). The rules match
  or beat them on kind, fan_out, gates and segmentation. On mandatory,
  Clef-flash reaches unit accuracy 0.85 against the old marker-only rule's
  0.62, and end to end 0.32 / 0.35 recall / precision. The current
  "unless conditional" rule reaches 0.43 / 0.32 with no call. Not adopted.
- **Haiku 5.5 as a routed labeler** (generative, offline replay of the
  research classifications, `research/system1-routing/route_haiku.py`):
  skeleton kind accuracy goes from 0.456 to 0.531, with 52 of 157 steps
  routed, for +$0.0026 per compile. Not adopted, because it is a generative
  model making a decision. That is the user's call.

## Low-confidence decisions, without an LLM decision

Below 0.7, a decision takes a conservative default and is listed in the
report's Open questions:

- data flow defaults to the step before (it can only over-order, never drop
  data)
- every node reads the request
- no signal means `pure_function`
- no named tool means a placeholder tool on the loop (it fails loudly at
  discovery)
- a server without a tool gets no MCP binding

## The agent proposes, never decides

After each of the agent's stops, rote restores every decision field, repair
rounds included, so repairs are judged on the pipeline that ships:

- the node fields `kind`, `inputs`, `fan_out`, `mandatory`, `constants`,
  `impl`, `mcp`, `tools`, `tool_servers`, `loop_body`, `signal`, `timeout`,
  `retry` and `source`
- the pipeline's `name`, `input`, `edges`, `entry_nodes`, `exit_nodes` and
  `config`

Added nodes are removed and dropped nodes come back. A `pipeline.yaml` the
agent leaves unparseable goes back to it for repair. On the last pass it is
kept for review and the skeleton is restored. That case came from a real
run, and it used to crash the compile.

Proposals go in `system1/proposals.yaml`. rote checks each one mechanically
(the node exists, the field is a decision, the IR still loads) and lists it
in the report. None is applied.

## Provenance

Briefs, the skeleton and the decisions live in the work dir, which is never
skill text, so a `# skill:` citation of them fails verification like any
made-up quote. Rule-table rows are cited verbatim from the skill's markdown.

## Spend for this build

$25.93 of the $30 cap:

| item | cost |
|---|---|
| v1 dev run | $1.12 |
| v1 bench | $6.11 |
| crashed ops-report run (estimated, its metrics were lost) | ~$0.80 |
| Sonnet 5.5 single run | $0.57 |
| main + #144 baseline bench | $7.00 |
| v2 bench | $2.26 |
| v2 expense-audit on Sonnet 4.6 | $0.81 |
| held-out, default path | $4.32 |
| held-out, `--system1` | $1.12 |
| Clef skeletons, 8 skills x 2 deciders, plus a 3-call spot check | $0.04 |
| fidelity compiles, rules and clef, expense-audit + ops-report | $1.78 |

The $0 hill-climbing loop is
`research/system1-routing/score_skeletons.py`. It scores skeletons with the
benchmark's own scorer, with no inference.
