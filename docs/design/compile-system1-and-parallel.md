# Faster, cheaper compile: a System-1 pass and parallel decomposition

Status: research, 2026-10-19. Nothing here changes the default compile
path. The prototypes live under `research/compile-system1/` and are not
imported by `rote`.

This document answers two questions the user asked:

1. **System 1.** Can a fast first pass (deterministic code, or a small
   cheap model) handle the mechanical parts of compiling a skill, so the
   expensive agent only does what needs it?
2. **Decomposition.** Can a skill be split into parts, compiled in
   parallel, and merged back into one valid IR?

It measures today's compile first, because the answer depends on where
the money and the minutes actually go.

## 1. How compile works today, measured

`rote compile` runs one agent (`claude -p`, Sonnet 4.6 by default) with
the rote-compile skill as its system prompt. The agent reads the rubric
(SKILL.md plus six references, about 2,400 lines), reads the source skill,
then writes `extracted/*.py`, `tests/*.py`, `signatures/*.py`,
`pipeline.yaml`, `eval.yaml` and `compile-report.md`. rote validates and
verifies the deliverable and sends failures back for up to two repair
rounds.

Seven real compiles from this week (`scratchpad/e2e/phaseb/out/*` and
`scratchpad/feat-cipf/final-*`), broken down with
`research/compile-system1/measure_runs.py`:

| run | wall | read + plan | write | repair | turns | repairs | output tok | cost | output share of cost |
|---|---|---|---|---|---|---|---|---|---|
| bdr-outreach | 27.4 min | 959 s | 458 s | 224 s | 63 | 2 | 115k | $4.46 | 39% |
| deal-monitor | 24.0 min | 822 s | 540 s | 79 s | 40 | 1 | 92k | $2.23 | 62% |
| expense-audit | 21.3 min | 744 s | 455 s | 81 s | 48 | 2 | 94k | $2.51 | 57% |
| invoice-push | 34.4 min | 1272 s | 641 s | 153 s | 41 | 2 | 151k | $3.72 | 61% |
| ops-report | 30.4 min | 1054 s | 538 s | 230 s | 68 | 2 | 144k | $3.54 | 61% |
| expense-audit (final) | 21.8 min | 795 s | 342 s | 170 s | 43 | 1 | 97k | $2.45 | 59% |
| ops-report (final) | 17.1 min | 408 s | 512 s | 106 s | 53 | 2 | 83k | $2.46 | 50% |
| **mean** | **25.2 min** | **865 s (57%)** | **498 s (33%)** | **149 s (10%)** | 51 | 1.7 | 111k | **$3.05** | 56% |

"read + plan" is everything before the first real Write. Inside it the
reads take about 30 seconds. The rest is one or two silent turns of 390 to
530 seconds each (the `top_gaps` column of the script). That is the agent
thinking through the whole compile before writing anything.

Four facts drive every option below:

- **Output tokens are the clock.** At roughly 75 output tokens a second,
  111k output tokens is about 25 minutes. Wall time is almost entirely
  generation, not tool calls or reads.
- **Most output is thinking, not artifacts.** The files a compile leaves
  behind total 110k to 135k characters, about 32k to 38k tokens. The other
  ~70% of output tokens are reasoning. The costcut lane measured the same
  thing directly (thinking is 62% to 76% of output) and found the big
  think often hits the output cap and is discarded.
- **Input is cheap.** Rubric reads are about 24k tokens, cached. Cache
  reads cost 11% to 42% of a run, and scale with turns times context,
  which is why BDR's 63 turns cost $1.89 in cache reads alone. PR #113
  found the compiler driver's first request is 70.7k tokens and would be
  11k with full context isolation (open follow-up there).
- **Every run repairs.** 1 to 2 repair rounds in all seven runs, 79 to 230
  seconds of repair per run.

So the savings available are in **reasoning the model does not need to
do**, then in fewer turns. A System-1 pass is worth something only if it
removes reasoning. Parallelism is worth something only if it splits the
reasoning across workers instead of repeating it.

## 2. What the System-1 experiments found

### (a) Deterministic SKILL.md extractor, zero inference

`research/compile-system1/system1/extract.py` (stdlib + PyYAML, 1 to 11 ms
per skill). It parses frontmatter and the heading tree, drops meta
sections (trigger phrases, routing tables, tool tables, limits), and
treats leaf procedural sections as steps, splitting further where the
skill itself enumerates sub-steps (bold-numbered paragraphs, numbered lists
with bold titles). Each step gets node proposals from the signals
node-kinds.md names, including its split rule (a tool call is a bare
`external_call`, and what is done to the result is a separate node). It also
records MANDATORY markers, MCP and tool mentions, cited references, and
every literal threshold in the prose. `assemble.py` turns the proposals
into a `rote.ir.Pipeline` that passes the real validator, so the
benchmark's scorer (`rote.bench.score.score_pipeline`) scores it exactly
as it scores the agent.

Four skills were read while writing the signals (a dev set). invoice-push
was held out: only its headings were seen, and the extractor was frozen
before it was scored.

Two metrics, because they disagree for a reason:

- **bench kind accuracy**: the benchmark's own metric. It pairs nodes by
  id and description token overlap, so it also measures naming.
- **section kind recall / precision** (`system1/score.py`): per SKILL.md
  section, the overlap of predicted and expected node-kind multisets. It
  ignores names and measures structure only.

| skill | expected nodes | extractor bench kind acc | extractor section recall / precision | agent bench kind acc | agent section recall / precision |
|---|---|---|---|---|---|
| bdr-outreach | 15 | 0.33 | 0.73 / 0.73 | (IR now invalid) | 0.80 / 0.44 |
| deal-monitor | 13 | 0.15 | 0.62 / 0.67 | 0.69 | 0.85 / 0.73 |
| expense-audit | 8 | 0.62 | 0.88 / 0.88 | 0.75 | 0.88 / 0.88 |
| ops-report | 11 | 0.82 | 0.91 / 1.00 | 0.82 | 0.91 / 0.91 |
| invoice-push (held out) | 13 | 0.31 | 0.62 / 0.33 | 0.77 | 0.92 / 0.67 |
| **mean** | | **0.45** | **0.75 / 0.72** | **0.76** (4 skills) | **0.87 / 0.73** |

"agent" is today's full Sonnet compile of the same skill. Its BDR IR no
longer loads under the current validator (#139 rejects one of its
inputs), so only the section metric covers it.

What the extractor gets right with zero inference:

- **Section structure.** It covered every expected section on the dev
  set and 10 of 11 on the held-out skill.
- **HITL gates.** 4 of 4 expected gates, plus one extra (BDR's "Confirm
  with the user" campaign brief, which the curated IR treats as pipeline
  input).
- **Literal constants.** 34 of 40 scalar constants in the curated IRs'
  `constants:` blocks are found verbatim (5/6, 5/5, 8/9, 13/16, 3/4).
  Precision is low (BDR yields 43 candidate numbers for 6 real ones), so
  these are evidence for the agent, not answers.
- **Rule tables to code.** `system1/rule_tables.py` lifts markdown tables
  into typed Python literals. On expense-audit the generated FX table is
  byte-for-byte equal to the hand-written `MONTH_END_RATES` (`==` returns
  True), and the category caps match the curated `category_caps_usd`
  values. BDR's limits table yields the three batch sizes (100, 250, 10)
  the curated IR uses. What it cannot know: that lodging is per night and
  the IR names it `lodging_per_night`, or that "no cap" on entertainment
  is a rule.
- **MANDATORY markers.** Every explicit marker is found, but markers cover
  only 5 of the 24 nodes the curated IRs flag mandatory. The rest are
  judgment (ops-report marks 10 nodes mandatory with no marker in the
  text), so mandatory recall is 0.11 for the extractor and 0.80 for the
  agent.

What it gets wrong, and why it is not a tuning problem:

- **Granularity.** How many nodes a step becomes (deal-monitor's "Search
  Gmail" is five curated nodes, invoice-push's steps 2 to 4 are one) is a
  design decision, not a text feature. This is most of the held-out miss:
  24 proposals for 13 expected nodes.
- **Loops whose body is the child headings** (invoice-push step 7) read
  as a container, not an `agent_loop`.
- **Names.** Heading slugs do not match the curated ids, which is why
  bench kind accuracy (0.45) sits far below section recall (0.75). Naming
  is cheap for any model and free for the agent.
- **Data flow.** It emits a document-order chain. Edge F1 is 0.70 on the
  matched nodes, but it knows nothing of `inputs:`, payload shapes,
  fan_out, or the pipeline input schema.

### (b) Small-model classifier

`system1/classify.py`: one schema-locked `claude -p --json-schema` call
per skill, with node-kinds.md in the prompt and rote's own context
isolation flags (imported from `rote.inference._runtime_helper`), so the
cost is the classifier's and not Claude Code's. Two modes: **hinted** (the
model sees the extractor's units and guesses and returns the node list per
unit) and **raw** (the model sees SKILL.md and references and returns
nodes with their source heading). Three Haiku trials per mode and skill,
one Sonnet 5.5 trial for reference.

| predictor | bench kind acc | section recall / precision | held-out bench kind acc | cost per skill | wall per skill |
|---|---|---|---|---|---|
| extractor (no model) | 0.45 | 0.75 / 0.72 | 0.31 | $0 | 1 to 11 ms |
| Haiku 5.5, hinted | 0.46 | 0.81 / 0.70 | 0.26 | $0.0026 | 7 to 27 s |
| Haiku 5.5, raw | 0.57 | 0.84 / 0.59 | 0.44 | $0.0029 | 11 to 41 s |
| Sonnet 5.5, raw (1 trial) | 0.59 | 0.77 / 0.69 | 0.54 | $0.054 | 5 to 13 s |
| today's agent (Sonnet 4.6, full compile) | 0.76 (4 skills) | 0.87 / 0.73 | 0.77 | $2.2 to $4.5 | 17 to 34 min |

Model ids and prices are from the claude-api skill (cached 2026-10-06):
Haiku 5.5 is $0.10 / $0.50 per million input / output tokens, Sonnet 5.5
$2 / $10. Costs are the CLI's `total_cost_usd`.

Findings:

- **Haiku is accurate enough for the kind label.** Sonnet 5.5 scores about
  the same (0.59 against 0.57 bench kind accuracy, better on the held-out
  skill, worse on section recall) at 20 times the price, so the cheapest
  model is the right one. Most of the remaining gap to the agent is
  granularity, naming and data flow, which a one-shot classifier does not
  attempt.
- **Confidence is a usable routing signal.** On matched nodes, Haiku's
  kind is right 97.5% of the time at confidence of 0.8 or more (79 of 81,
  hinted) and 54% below it (7 of 13). Raw mode: 96% vs 71%. Sonnet 5.5:
  31 of 32 vs 4 of 6. The
  extractor's own heuristic confidence carries no signal (83% vs 88%). A
  router can trust Haiku's high-confidence labels and escalate the rest.
- **The hint helps structure but hurts naming.** Hinted mode inherits the
  extractor's sections (better section precision) and its splits (worse
  on the held-out skill).

Inference spend for this research: (a) $0, (b) $0.35, the planner dry
runs $0.04, (c) $5.21, the Option 1 run $3.11. Total $8.71.

### (c) Parallel decomposition on one skill

`parallel/parallel_compile.py` splits expense-audit (7 steps) into three
parts with the extractor (steps 1 to 3, 4 to 5, 6 to 7), asks one Sonnet 5.5
call for the cross-part data contract ($0.011, 7 s), writes each part as a
sub-skill (original frontmatter and preamble, a contract block, that part's
sections verbatim, all references), runs three real `rote compile`
processes concurrently from an installed wheel, merges, and verifies the
merge against the original skill. One run.

| | single agent (baseline) | 3 parallel parts |
|---|---|---|
| wall time | 1,306 s (21.8 min) | 983 s (16.4 min), slowest part 975 s |
| cost | $2.45 | $5.21 (planner $0.01, parts $1.52, $1.66, $2.02) |
| turns | 43 | 90 (30 per part) |
| output tokens | 97k | 186k (54k, 59k, 73k) |
| read + plan before first write | 795 s | 561 s, 481 s, 672 s |
| bench node recall / kind accuracy | 0.75 / 0.75 | 0.75 / 0.75 |
| bench edge F1 / wave order | 0.25 / 0.87 | 0.00 / 0.93 |
| mandatory precision / recall | 0.57 / 0.80 | 0.50 / 1.00 |
| generated tests | 84 passed | 143 passed |
| emit dbos and cloudflare | ok | ok |
| verification | passed | **failed: 2 citation errors** |

The baseline is `scratchpad/feat-cipf/final-expense-audit`, scored with the
same collector (`bench_row.py`, which calls `rote.bench.collect.collect_job`).

The finding that matters most: **a part with 2 or 3 steps still spends 480
to 670 seconds thinking before its first write, against 795 seconds for the
whole 7-step skill.** Most of the big upfront think is a fixed cost of
taking on the rubric, not proportional to the skill. Splitting into three
full compiler agents pays that cost three times. Output tokens nearly
doubled, cost more than doubled, and wall time fell only 25% because the
slowest part still carries the whole fixed cost.

The merge itself worked mechanically. All 12 cross-part inputs were rewired
from `pipeline.input.<field>` to the producing node's output, the merged
IR loads, every module is implemented, and both emitters accept it. The
problems it surfaced, in order of how much they would cost to fix:

1. **Contract text leaks into provenance.** Two modules cite the contract
   block as skill text (`extracted/a_checks.py` quotes the planner's field
   description). Each part's own verification passed, because its sub-skill
   contains the contract. Only verification against the original skill
   catches it, so a real implementation needs a post-merge repair round,
   with the contract kept outside anything citable.
2. **Parts disagree on shared shapes.** All three parts declared
   `expense_lines` and `period` with different JSON Schemas (one used
   `$defs`, two inlined). The merge keeps the first and reports the rest.
   An earlier version of the prototype dropped `$defs` and left a dangling
   `$ref`. That is fixed, and the re-merge cost no inference.
3. **Parts drift from the contract.** Part B exported `employee_audits`
   and `employee_totals` where the contract said `employee_reports`, and
   part C read `converted_lines` straight from part A. Both still resolved,
   because the merge rewires by output key, but a field no earlier node
   produces would be an unresolved input, and there is no agent left to ask.
4. **Local judgment, global flags.** Every part marked all of its nodes
   mandatory (10 of 10, against 5 of 8 curated), and part C prefixed its
   node ids with `c_` after reading the file-prefix instruction. Neither
   breaks the IR. Both lower quality.
5. **Shared code cannot be shared.** The curated expense pipeline has
   `audit.py` import helpers from `fx.py` and `checks.py`. Parts cannot see
   each other's modules, so each reimplemented what it needed.

### Option 1 measured: today's compile plus a System-1 brief

`system1/brief_compile.py` builds a brief (the extractor's units, one
Haiku 5.5 classification with confidences, the literal thresholds, the rule
tables already lifted to Python), puts it in a copy of the skill as a
reference file, and runs an unmodified `rote compile` on that copy. The
brief cost $0.0008 and 8 seconds. One run, so treat the direction as a
signal, not a measurement of the mean.

| | baseline (2 runs) | with System-1 brief |
|---|---|---|
| wall time | 1,280 s and 1,306 s | 1,286 s |
| read + plan before first write | 744 s and 795 s | **368 s** |
| repair time | 81 s and 170 s | **471 s** (2 rounds) |
| cost | $2.51 and $2.45 | $3.11 |
| output tokens | 94k and 97k | 102k |
| turns | 48 and 43 | 50 |
| bench node recall / edge F1 | 0.75 / 0.25 | 0.88 / 0.55 |
| mandatory precision / recall | 0.57 / 0.80 | 0.71 / 1.00 |
| verification | passed | passed (after 2 repairs) |

The brief did what it was designed to do: the upfront think halved. The
time came back in repair. The first deliverable had 5 contract problems, 5
provenance gaps and failing tests. The reasoning did not disappear. It
moved from planning, where it is one long turn, to repair, where every turn
re-reads the conversation. Structure got better (more nodes matched,
double the edge F1, better mandatory flags), which is what the brief
carries. First-pass code got worse, which is what it does not carry.

The bench reports kind accuracy 0.62 for this run against 0.75 for the
baseline. Both "misses" are its name matcher pairing `save_audit_report`
with the curated `audit_report` node, not wrong kinds.

## 3. What is global and what is local

A part of a skill can be compiled alone only if everything it shares with
the rest is decided first. From the merge in (c), the shared things are:

| concern | global or local | why |
|---|---|---|
| pipeline input schema | global | every part reads it, and two parts naming the same field differently breaks the merge |
| cross-part data fields | global | the producer's `output` key and the consumer's `inputs:` reference must use one name and one shape |
| node ids | global namespace, local choice | parts pick ids, and the merge must detect collisions and rewrite every reference |
| file names under `extracted/`, `tests/`, `signatures/` | global namespace | two parts writing `extracted/report.py` is a silent overwrite, so the prototype prefixes per part |
| shared types and helpers (a `Finding` shape, `to_cents`) | global | the curated expense IR has `audit.py` import from `fx.py` and `checks.py`, and parts cannot see each other's code |
| edges, waves, entry/exit nodes | derived after merge | recomputed from the rewired `inputs:` |
| `source.section` | local | each node's section is in its own slice, and slices keep headings verbatim so provenance still points at the original SKILL.md |
| MANDATORY | local flag, global rule | set per node, but every worker marked all of its nodes mandatory (10 of 10). It needs one rule applied to the whole pipeline |
| node bodies, tests, judge signatures | local | the bulk of written output, and the part that parallelizes |
| `compile-report.md` open questions | merged | concatenated per part. Cross-part questions (a field one part assumed and another never produced) appear only after the merge |
| `eval.yaml` | merged | per-step estimates concatenate |
| verification (citations, tests, emit) | global, after merge | run once against the original skill |

The contract (row 2) is the hard part: someone has to decide the cross-part
fields before the workers start. In the prototype one cheap Sonnet 5.5 call
writes it ($0.01 to $0.02). If the contract is wrong, every downstream
failure looks like a worker bug.

## 4. Options

All estimates use the measured baseline (mean $3.05 and 25 minutes, range
$2.2 to $5.1 and 20 to 50 minutes across skills) and the decomposition
measured above (output tokens set the clock at about 75 a second, and about
70% of output is reasoning). The costcut lane's effort and work-order
changes attack the same upfront think, so their savings overlap with
Options 1 and 2 and will not simply add. Options 3 and 4 change the
structure of the run and stack more cleanly.

| | Option 1: System-1 brief | Option 2: System-1 owns the skeleton | Option 3: router plus per-node workers | Option 4: parallel full compiles |
|---|---|---|---|---|
| what changes | a deterministic + Haiku brief goes into the work dir. Agent and rubric unchanged | System-1 writes the node skeleton, constants, rule-table modules and the `eval.yaml` skeleton. The agent gets a smaller rubric (implementation, judges, data flow) and the job "decide the open sections, implement and wire" | System-1 classifies. Gates, bare MCP calls and table-driven checks are templated. Each other node is one non-agentic call (module + tests, or a judge signature). One integrator agent writes data flow and the report | the skill is split per section group, one full compile per part, contract first, merge after |
| measured | yes, 1 run (expense-audit) | its parts (extractor, rule tables, Haiku, the upfront-think share) | its parts (Haiku routing, Sonnet 5.5 call cost and speed) | yes, 1 run (expense-audit) |
| cost vs $3.05 mean | +27% measured. Estimate 0% to +10% after tuning | estimate -15% to -30% ($2.1 to $2.6) | estimate -30% to -50% ($1.5 to $2.1) | +112% measured. Estimate +40% to +70% with Option 2's rubric per worker |
| time vs 25 min mean | 0% measured | estimate -15% to -30% (18 to 21 min) | estimate -50% to -65% (9 to 12 min) | -25% measured. Estimate -35% to -45% with a smaller rubric |
| quality | structure up (node recall 0.75 to 0.88, edge F1 0.25 to 0.55), first-pass code down (2 repairs) | structure as Option 1, code depends on keeping enough rubric | highest risk: shared helpers and types across nodes, cross-node tests, consistent naming | structure unchanged, flags worse, provenance leak, shapes drift |
| hallucinated merges | none (one agent) | none (one agent) | the integrator must reconcile N independent modules | contract drift, shape conflicts, id prefixes |
| broken data flow | none | low: the agent still owns `inputs:` | medium: the integrator wires code it did not write | medium: rewiring by output key, and an unresolved field has nobody to ask |
| provenance | the brief must stay outside citable text (in the work dir) | same, and System-1's constants carry their own `# skill:` quotes | per node, easy | contract text was cited twice in one run |
| complexity | small: about 300 lines, the driver writes one more file | medium: driver prompt, rubric split, skeleton writer, bench gates | large: new orchestrator driver, contract format, per-node repair routing | medium: slicer, planner, merger, post-merge repair |
| IR and invariants | no IR change. Invariant 2 holds (one more file in the work dir) | no IR change. Invariant 2 holds | no IR change if it is a driver that writes `pipeline.yaml`. Invariant 4 needs MANDATORY set deterministically, not per worker | no IR change. Invariant 4 at risk: workers over-mark and could under-mark |

How the estimates are built. The measured facts: about 57% of wall time is
the upfront think, and 60% to 80% of that think is fixed per agent (parts
with 2 to 3 steps think 480 to 670 s, the whole skill 745 to 795 s).
Output runs at about 75 tokens a second. A Sonnet 5.5 call over a whole
skill costs $0.03 to $0.11 and returns in 5 to 13 seconds. Haiku's
high-confidence labels are 96% to 98% right.

- Option 2 assumes the fixed think shrinks in proportion to the rubric the
  agent must hold (from about 2,400 lines to about 900), and keeps
  Option 1's repair penalty in the range.
- Option 3 prices 8 to 15 per-node calls at about 8k input and 4k output
  tokens on Sonnet 5.5 ($0.05 to $0.06 each), an integrator at a third of
  today's agent, and one repair round.
- Option 4's tuned estimate applies Option 2's rubric saving to each
  worker.

These are estimates, not measurements. The Option 1 run is the reason to
distrust any estimate that assumes reasoning disappears rather than moves.

## 5. Recommendation

Do not change the default path, and do not pursue Option 4 as built. The
one run says splitting a skill across full compiler agents doubles the
cost to save a quarter of the time, because the expensive part of a compile
is a fixed think per agent, not per step.

Build Option 2 next, behind an experimental flag, and judge it with
`rote bench` on all five examples against the costcut lane's new baseline.
Its pieces exist and are measured: the extractor (structure at zero cost),
`rule_tables.py` (exact constants), and Haiku as the router (calibrated
confidence for $0.003 a skill). The Option 1 run shows what Option 2 has to
fix. Handing the agent a plan cuts upfront thinking, but the agent still
needs the rubric's implementation guidance to get code right the first
time. So Option 2 moves the structural rubric (classification,
crystallization, IR assembly) into System-1 and keeps `implementation.md`
and `llm-judge-extraction.md` in front of the agent.

Keep Option 3 as the follow-on if Option 2 confirms that the fixed think
scales with the rubric. That is the one measurement Option 3 depends on,
and it is cheap to take: one Option 2 run per example skill, about $12.

Separately, worth one bench run in the costcut lane: Sonnet 5.5 costs
$2 / $10 against Sonnet 4.6's $3 / $15, and answered whole-skill
classification in 5 to 13 seconds.

## 6. What you need to decide

1. **Whether to fund Option 2** as an experimental `--system1` flag:
   about a day of build, then about $12 of bench runs.
2. **Haiku in the compile path, or deterministic only.** Haiku adds $0.003
   and 10 to 40 seconds per compile, through the credential compile already
   uses. Without it, routing loses its only calibrated confidence (the
   extractor's own confidence carries no signal).
3. **Whether System-1's node splits are binding or advisory.** Binding
   saves the most thinking but anchors the agent on the extractor's
   granularity, which was 0.33 precise on the held-out skill. Advisory is
   safer and saves less.
4. **Whether a faster but costlier compile is ever wanted.** If rote cloud
   compiles are judged on time to first pipeline rather than tokens,
   Option 4 (or Option 3) becomes a product choice, not a cost one.
5. **A provenance rule for generated hints.** Confirm that briefs and
   contracts live in the work dir and are never citable as skill text. The
   parallel run shows agents will cite them otherwise.

## Appendix: reproducing the numbers

The raw outputs behind every table are in `research/compile-system1/results/`:
`bench-eval.jsonl` (every predictor and trial through the bench scorer),
`classifier-spend.jsonl` (each classifier call's cost and wall time),
`parallel-contract.json`, `parallel-expense-audit.run.json` (the original
run, with its 983 s end-to-end wall) and `parallel-expense-audit.json` (the
re-merge after the `$defs` fix, which spent no inference), and
`brief-expense-audit.{md,json}` (the brief the agent saw and that run's
metrics).

All commands run from the repo root with rote state isolated (every
`ROTE_*_PATH` override from `tests/conftest.py`), and compiles run from a
wheel installed in a scratch venv.

```sh
# today's runs, broken down
python research/compile-system1/measure_runs.py <compile --out dir> ...

# (a) extractor on one skill, and the section metric over all five
python research/compile-system1/system1/extract.py examples/expense-audit/skill
python research/compile-system1/system1/score.py
python research/compile-system1/system1/rule_tables.py examples/expense-audit/skill

# (b) classifier trials, then every predictor through the bench scorer
python research/compile-system1/system1/run_classify.py <state> <runs> raw claude-haiku-5-5 \
  bdr-outreach deal-monitor expense-audit ops-report invoice-push
python research/compile-system1/system1/bench_eval.py <runs> <agent compile root>

# (c) parallel compile of one skill
python research/compile-system1/parallel/parallel_compile.py --rote <venv>/bin/rote \
  --skill examples/expense-audit/skill --parts 0-2,3-4,5-6 --work <dir> --state <state>

# Option 1 measured
python research/compile-system1/system1/brief_compile.py --rote <venv>/bin/rote \
  --skill examples/expense-audit/skill --work <dir> --state <state>

# any compile output through the benchmark's collector
python research/compile-system1/bench_row.py <out dir> <skill dir> <expected pipeline.yaml> <wall s>
```
