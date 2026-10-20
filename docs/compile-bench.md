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

`rote bench` is hidden from `rote --help` because it is a developer tool.
It needs the repo's `examples/`, which the wheel does not ship. The
options:

| Flag | Meaning |
| --- | --- |
| `--skills a,b` | Subset (default: expense-audit, ops-report, invoice-push, deal-monitor, bdr-outreach) |
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
only, no compiled outputs.

Diff against it with `--compare`, or print it with
`rote bench --report examples/bench/baseline.json`. When a change lands
that moves the numbers on purpose, re-run the set and replace the file in
the same PR.
