# The compile benchmark (`rote bench`)

`rote bench` makes compile quality, cost and speed measurable, so a rubric
or architecture change is judged by numbers instead of by reading a few
pipelines. It compiles example skills with the real compiler, scores each
compiled `pipeline.yaml` against the hand-written
`examples/<name>/expected/pipeline.yaml`, and writes a `results.json` that
a later run can be diffed against.

It spends tokens, so it is an **eval**, run by hand. It is never part of
`pytest tests/`. Only the scorer is tested there
(`tests/test_bench_score.py`, fixture IR pairs with hand-worked scores).

## Running it

Run it from an immutable wheel install of the branch you are judging, in
an empty working directory. Never run it from the editable tree: a
compile that takes 10 minutes should measure one fixed version of rote.

```sh
uv build --wheel --out-dir /tmp/bench/wheel
uv venv --python 3.11 /tmp/bench/venv
uv pip install --python /tmp/bench/venv/bin/python "/tmp/bench/wheel/rote_cli-*.whl[dbos,mcp,api,temporal]"
mkdir -p /tmp/bench/cwd && cd /tmp/bench/cwd

# The fast loop: one small skill, about $2.5 and 10 minutes.
/tmp/bench/venv/bin/rote bench --examples ~/rote/examples --skills expense-audit \
  --out /tmp/bench/run-ea --label "$(git -C ~/rote rev-parse --short HEAD)" \
  --compare ~/rote/examples/bench/baseline.json

# All five skills, with a spending cap.
/tmp/bench/venv/bin/rote bench --examples ~/rote/examples --budget 20 --out /tmp/bench/run-all
```

The held-out set lives in `examples/holdout/` and is benched separately,
with `--examples ~/rote/examples/holdout`. Nobody tunes the compiler
against it (see [its README](../examples/holdout/README.md)), so a
change that lifts the five examples and drops the holdout scores is
overfitting. Discovery on `examples/` never reaches it.

`rote bench` is hidden from `rote --help` because it is a developer tool.
It needs the repo's `examples/`, which the wheel does not ship. The
options:

| Flag | Meaning |
| --- | --- |
| `--skills a,b` | Subset (default: every `<name>/` under `--examples` with a `skill/SKILL.md` and an `expected/pipeline.yaml`. For `examples/` that is expense-audit, ops-report, invoice-push, deal-monitor, bdr-outreach, in that order. Other skills follow, smallest first) |
| `--repeat K` | Compiles per skill, for run-to-run variance |
| `--concurrency N` | Compiles in flight at once (default 1) |
| `--budget USD` | Stop launching compiles once total `total_cost_usd` reaches this. Running ones finish, so the overshoot is at most `N - 1` compiles |
| `--model`, `--agent` | Passed to `rote compile` |
| `--runtime` | The compile's `--runtime` (default `dbos`, the logged-out default) |
| `--no-tsc` | Skip `npm install` + `tsc --noEmit` on the Cloudflare emit |
| `--compare OLD.json` | Print this run's deltas against an older results.json |
| `--resume` | Keep the jobs already scored in `<out>/results.json` and run the rest. A job whose compile produced no pipeline (a usage limit, a crash) is re-run, and what it spent stays in `spent_usd` under `replaced_jobs`. Everything already spent counts toward `--budget` |
| `--rescore RUN_DIR` | Re-collect an existing run's compiles with the current scorer (needs `--examples`). No tokens |
| `--report R.json` | Print a results.json (with `--compare`, just the deltas). No tokens |
| `--fidelity` | Also run each compiled pipeline on its example's fidelity cases (see below). Judges hit a real model, so this spends tokens per compile |
| `--record-reference` | Run each skill as an agent on its fidelity cases, `--trials` times (default 3), and write `examples/<name>/fidelity/reference.json`. No compile. Spends tokens |
| `--fidelity-of DIR` | Score one compiled dir (`pipeline.yaml` + `extracted/`) on the cases of the single `--skills` example and print JSON. For checking a hand-edited or broken pipeline |

Each job runs `python -m rote compile <skill> --local --runtime dbos
--no-deploy --no-eval --json --out <run>/<skill>/r<k>` through the same
interpreter as the bench. Two things are forced, not configurable:

- **State isolation.** Every `ROTE_*_PATH` override the test suite sets
  points at `<run>/state/`, and `ROTE_RUNTIME`/`MODEL`/`AGENT`/`DEPLOY` are
  dropped. A personal config could pin a runtime or a model, and a cloud
  login would move the compile server-side.
- **No answer key.** The skill is copied to `<run>/<skill>/r<k>/skill-src/`
  first, so the compiler agent cannot step from the skill dir into
  `expected/`.

## What it reports

Every job is checked the same way, whatever exit path its compile took:
the scorer reads `compiled/` (or `compiled.rejected/`) and re-runs rote's
own offline checks on it.

| Column | Meaning |
| --- | --- |
| `nodes` | compiled / expected node count |
| `nodeR`, `nodeP` | expected nodes matched / compiled nodes matched |
| `kind` | expected nodes whose matched compiled node has the same kind. A missing node counts as wrong. The full confusion map is in the JSON |
| `kmix` | matching-free kind agreement: shared per-kind node counts over the larger node count. Read it beside `kind` (see below) |
| `edgeF1` | F1 of dependency edges (declared edges plus `inputs` references) between matched nodes |
| `waves` | pairwise ordering agreement: for each pair of matched nodes, do both pipelines put them before / together / after? Robust to a whole wave shifting |
| `fanP`, `fanR` | fan_out precision and recall vs expected |
| `mandP`, `mandR` | mandatory precision and recall |
| `gates`, `sig`, `gateJ` | compiled/expected gates, gates with the same signal name, mean Jaccard of the resume payload keys |
| `inR`, `inP` | pipeline input field names, recall and precision |
| `impl` | pure_function nodes implemented / total (the rest are stubs) |
| `tests` | the generated tests, run in the OS sandbox |
| `citeE`, `assum`, `OQ` | citations that are not in the skill, `# rote: assumption:` markers, Open questions entries in compile-report.md |
| `ok` | rote's verification verdict on the deliverable |
| `exit`, `rep`, `turns` | compile exit code, repair rounds, agent turns |
| `cost$`, `inTok`, `outTok` | `total_cost_usd` and tokens. `inTok` includes cache writes and reads, which are nearly all of it |
| `wall_s` | wall time of the compile subprocess |
| `dbos`, `cf`, `tsc` | does the IR emit on dbos and cloudflare, does the Cloudflare app type-check |

`results.json` is portable: job dirs are relative to the run dir, and the
run dir, the examples checkout and the home directory are replaced by
`<run>`, `<examples>` and `~` everywhere else, so it can be committed.
Per-phase seconds (from the compile's `progress.jsonl`), the matches,
missing and extra node ids, and the per-gate rows are in `results.json`.
The `AGGREGATE` row is a mean of each column (cost and wall time are
sums). A rate with nothing to find, such as fan_out recall on a skill with
no fan_out, is `-` and is left out of the mean rather than counted as 0
or 1.

### How nodes are matched

An expected node pairs with the compiled node of the same id. Otherwise
it pairs with the most similar compiled node: the mean of the id-token
Jaccard and the (id + description)-token Jaccard, kept at 0.25 or above.
Pairs are taken greedily, highest similarity first, one to one. The
matches are listed in the JSON, so a surprising score can be checked by
reading them.

Lexical matching misfires when a compile splits or merges steps. In the
baseline, expense-audit's expected `file_report` (render and save, an
external_call) became two compiled nodes, `render_audit_report` and
`save_audit_report`, and the two `audit report` nodes matched crosswise,
which counts two kind errors that are really one split. That is why
`kmix` exists: it cannot misfire this way, and a `kind` drop with a
steady `kmix` usually means a matching artifact, not a classification
regression. Check `structure.matches` before acting on a kind change.

The expected pipelines are hand-written, so they are one good answer, not
the only one. A compile that splits a step in two scores an extra node,
and a reasonable rename can fall under the threshold. Compare runs with
each other rather than chasing 1.0.

## Behavioral fidelity

The structural scores compare one IR with another. Fidelity asks the
question that matters more: given the same input, the same tool results
and the same answers at each gate, does the compiled pipeline do what the
skill does?

An example opts in with `examples/<name>/fidelity/cases.yaml`:

- **An output contract**: the fields a run's result must carry (totals,
  ids flagged, the review queue, the report status). The skill side is
  told to write them, and the pipeline side is matched against them by value.
- **Fixture MCP servers**, when the skill uses MCP: tool names and
  schemas, plus per-case responses. `rote.bench.fixture_mcp` serves them
  over Streamable HTTP. Both sides talk to the same fixture, so they see
  byte-identical results, and the fixture's call log (not a transcript)
  is the tool trace for both.
- **Two or three cases**: an input (or a patched copy of the example's
  input), the fixed payload a person gives at each gate, and the fixture
  responses. Each case has several records and crosses different
  branches: a clean batch, a batch with new findings, an input the skill
  must stop on.

**The reference** is the skill itself. `--record-reference` runs it
through the baseline path (`claude -p`, the prompt `rote baseline` uses,
plus the output contract and the gate answers) on every case, several
times, and commits `reference.json`: each run's output and tool calls,
and what all runs agree on. Agents are not deterministic, so **only stable
fields count against a pipeline**: a field whose normalized value differs
between two agent runs of the same case is listed under
`unstable_fields` and skipped. Prose fields fall out this way on their
own. The reference is a one-off cost, paid again only when the cases
change.

**The pipeline side** emits the compiled IR for DBOS, points each MCP
server it needs at the fixture (by name or a listed alias, through
`ROTE_MCP_<SERVER>_URL`), sends each gate's payload in gate order, runs
the case, and reads every step's output back from the DBOS system
database. DBOS, not plain python, because it persists every step: a
pipeline that dies at its last node still shows what it computed. Judges
call a real model.

Each case scores five components, each in [0, 1], and their mean (a
component with nothing to compare is left out):

| Component | Meaning |
| --- | --- |
| `outputs` | stable reference fields the pipeline reproduced. Matched by value anywhere in the pipeline's step outputs, because key names are the compiler's choice. Lists compare as sets, and a list of records per field (the set of `expense_id`s in the review queue). A value that could match by coincidence (an integer under 10, a boolean, a short string, an empty list) also needs a key sharing a word with the field. Gate payloads received are excluded, so a pipeline gets no credit for echoing the approver's answer |
| `outcome` | the run completed when the skill completed, and stopped when the skill stopped |
| `reads` | reads every reference run made, recall, averaged with the share of the pipeline's reads that some reference run also made. A read is identified by the fixture response that answered it, so a differently worded search for the same data still counts |
| `side_effects` | writes, in order (LCS over the longer sequence) when the reference runs agree on the order, as a set otherwise |
| `gates` | key overlap between the case's gate payload and what the compiled gate declares |

`fidelity` is the mean case score. It is a top-level column (`fidel`) in
the table when `--fidelity` is on, with its judge cost under
`fidelity_cost_usd`, apart from the compile cost.

### Checking that the metric measures behavior

The gold pipeline (expense-audit's hand-written `expected/`) and two
broken copies, scored with `--fidelity-of`:

| Pipeline | fidelity | base | second_batch | unknown_currency |
| --- | --- | --- | --- | --- |
| gold | 0.93 | 1.00 | 0.91 | 0.89 |
| mandatory duplicate check dropped | 0.86 | 0.94 | 0.87 | 0.78 |
| `duplicate_ids` and `late_ids` inputs swapped | 0.66 | 0.58 | 0.51 | 0.89 |
| the pre-#144 baseline compile | 0.17 | 0.00 | 0.00 | 0.50 |

Each break costs what it should: the dropped check loses the duplicate
id and the review-queue ids, and the swap crashes the report (so it
loses `outcome` too) except on the case that stops before the checks
matter. The baseline compile renamed the pipeline input (`lines`,
`fx_rates` instead of `period`, `submissions`), so it cannot run on the
skill's own input at all. The structural score had the input recall at
0.5 and nothing else alarming.

The gold misses four fields on second_batch because the skill's own
agent runs, three of three, file EXP-2410 (software over its cap) as a
block, where the hand-written pipeline treats a cap breach as review. The
reference is the skill as it runs, so that counts against the gold too.
The base case shows the agent's other face: its three runs disagree on
`reimbursable_usd` ($2,990.98, $2,414.34, $2,544.73), so that field is
unstable there and not scored.

Costs: recording the references took $6.01 once (expense-audit $4.51
for 9 runs, ops-report $1.50 for 6). Per-compile fidelity on these two
examples costs nothing, since neither compiled pipeline has a judge. A
pipeline with judges pays for its judge calls, reported as
`fidelity_cost_usd`.

Coverage: expense-audit (3 cases, no MCP) and ops-report (2 cases, Drive
and Gmail fixtures). invoice-push drives a browser, which a fixture
server cannot stand in for. deal-monitor reads a second skill from the
user's home directory (`~/.claude/skills/deal-scoring`) that the repo does
not have. bdr-outreach writes to five systems and is the most expensive
skill to run as an agent. None of the three has cases yet.

## The committed baseline

[`examples/bench/baseline.json`](../examples/bench/baseline.json) (and its
table, `baseline.txt`) is one repeat on origin/main `3dc2e16`, the 0.15.0
release, with the default compiler model. It **predates #144** (compiler
effort low), so it is the reference for "before #144", not for current
main. A same-day bench of main plus #144 replaces it as the reference.

It covers four skills: expense-audit, ops-report, invoice-push and
deal-monitor. bdr-outreach is not in it. Its first attempt hit the weekly
usage limit after $2.19, and it was not re-run, to save money. The run
spent $15.01 in all: $10.57 on the four scored compiles, plus $2.26
(a deal-monitor attempt) and $2.19 (bdr-outreach), both killed by the
usage limit. Those two are listed under `replaced_jobs`. It holds metrics
only, no compiled outputs. It was rescored with `--fidelity` (no compile,
no tokens): expense-audit 0.17 and ops-report 0.59. The ops-report compile
reads every source right, then crashes because its gate wants
`shipper_brand` in each dwell row where the duty manager's answer, worded
as the skill words it, says `shipper`.

Diff against it with `--compare`, or print it with
`rote bench --report examples/bench/baseline.json`. When a change lands
that moves the numbers on purpose, re-run the set and replace the file in
the same PR.
