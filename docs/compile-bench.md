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
| `--decider D` | Passed to every `rote compile --decider`, to pin the decision model. Unset, each compile takes `auto`, which depends on the credentials where the bench runs; the `chk` column says which ran |
| `--compile-flag F` | Passed to every `rote compile`, repeatable. Compiles run System-1 by default, so `--compile-flag=--no-system1` benches the full-rubric path. Older rows below that say `--system1` were measured when it was opt-in |
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
- **The example's MCP tools.** An example whose fidelity cases use
  fixture MCP servers bundles each server's `tools/list` in the skill dir,
  at `skill/mcp-tools.json`. A plain compile with nothing registered
  grounds its tool arguments in it (a live `tools/list` of a registered
  server wins), and the fixture servers serve exactly those tools:
  `fidelity/cases.yaml` adds only what a fixture needs beyond the
  snapshot, the server's `aliases` and each tool's `side_effect`, and the
  bench refuses to start when the two name different servers or tools.
  So the bench registers nothing for the compile, as on a first run, and
  the schemas the compile reads are the ones the fixtures enforce when
  the pipeline runs.
- **The example's sample run.** An example with fidelity cases bundles
  its sample run in the skill dir, at `skill/sample-input.json`, so the
  bench's copy carries it and the compile finds it exactly as a plain
  `rote compile examples/<name>/skill` does. It is the
  `{"input", "signals"}` envelope a user would supply, with the fidelity
  cases' keys and none of their values (a test pins both). rote extracts
  the input contract and gate payload keys from it and runs the
  pipeline's deterministic steps on it before the compile is done
  (`rote.compiler.grounding`, `rote.compiler.prefix`). `input.json`,
  `eval-input.json` and `fidelity/` sit beside the skill, not in it, so
  the compile never sees them.

## Benching other models through Cloudflare AI Gateway

The `openai-api` driver reaches any model on the Cloudflare catalog through
the account's OpenAI-compatible route, billed by Unified Billing (third-party
models) or Workers AI (`@cf/` models). No code or provider key is needed,
only two variables. The token must belong to the account you mean to bill:
check `GET /accounts/<ACCT>` names it before spending.

```sh
export OPENAI_BASE_URL="https://api.cloudflare.com/client/v4/accounts/<ACCT>/ai/v1"
export OPENAI_API_KEY="<cloudflare api token>"
ROTE_EFFORT=low /tmp/bench/venv/bin/rote bench --examples ~/rote/examples \
  --skills expense-audit --agent openai-api --model @cf/deepseek-ai/deepseek-v4.1-flash-1017 \
  --out /tmp/bench/run-dsv --compare ~/rote/examples/bench/baseline.json
```

`--effort` (or `ROTE_EFFORT`) reaches the wire as `reasoning_effort`. Leave
it unset and the endpoint uses the model's own default, which for current
open-weight reasoning models is usually the top level. The bench has no
price table for these models, so `cost$` is empty: price the run from its
token counts, or from the gateway's logs, which record each call's cost.

Measured 2026-10-19, one expense-audit compile each, effort low, against
main + #144 on Sonnet 4.6 ($1.00, 362 s, kind 0.75, edgeF1 0.80, verified):

| Model | $ | wall | verified | kind | edgeF1 | note |
| --- | --- | --- | --- | --- | --- | --- |
| `@cf/deepseek-ai/deepseek-v4.1-flash-1017` | 0.24 | 862 s | yes | 0.88 | 0.40 | ops-report: $0.19, 538 s, kind 0.73 |
| `@cf/moonshotai/kimi-k2.7-code` | 0.64 | 920 s | yes | 0.88 | 0.15 | ops-report: not verified |
| `@cf/zai-org/glm-5.3-flash` | 0.06 | 460 s | no | 0.62 | 0.75 | 7 red tests after 2 repairs |
| `openai/gpt-oss-120b` (Groq) | <0.05 | 16-27 s | no | - | - | invalid IR, or a malformed tool call |
| `google/gemini-3.8-flash` | 0.08 | - | no | - | - | stopped at turn 6 with no pipeline |
| `openai/gpt-6-luna` | - | - | no | - | - | the route rejects tools with reasoning |

Behavioral fidelity (`rote bench --fidelity-of`, running the compiled
pipeline on the skill's own input) does not separate them on expense-audit:
every compile, Sonnet's included, scores 0.11 to 0.17 there because no
compile names the input fields the way the bundled input does. The
model-specific defects under that are real, though: DeepSeek demands an
`fx_rates_markdown` input the skill never takes, and Kimi ships a
`load_fx_rates` stub. On ops-report DeepSeek scored 0.50 and Kimi 0.38,
against 0.38 to 0.75 for Sonnet 4.6 compiles, with a fixture caveat on the
Gmail and Drive result shapes.

On the System-1 path (rules decide, the model writes modules and
text) on expense-audit: gpt-oss-120b on Groq ran in 31 s at low effort and 67 s
at medium, but stubbed 7 of 7 and 1 of 7 modules and did not verify (fidelity
0.17 and 0.11). DeepSeek wrote 0 of 7 in 454 s for $0.12. At high effort
gpt-oss called a tool that does not exist and Groq rejected the turn.

So no cheap model is a default or an opt-in for the compile agent yet: the two
that verified ran at 2.4 times Sonnet's wall time with worse edge F1, and
the fast one does not write working code. Cerebras is not on Unified Billing.
Groq is, with gpt-oss only.

### Decision models for System-1's choices

Cloudflare's Clef (`@cf/cloudflare/clef`, `clef-flash`) and Typesafe's Jev
(`typesafe/jev-1.14.0`) are not text generators. They take a state and typed
questions (`noul` yes/no, `choice`, `score`) and return a calibrated
probability per answer, for $0.04 to $0.24 per million input tokens with
output free, in 0.3 to 0.7 s a call. They fit the rule that code, not a
language model, makes compile decisions. Measured on System-1's extractor
units, one call per step, phrasings and thresholds fixed on the 5 dev
examples and then applied unchanged to `examples/holdout/` (23 steps):

| Decision | Rules dev / holdout | Best model dev / holdout |
| --- | --- | --- |
| Node kind (per-step kind-set Jaccard) | 0.77 / 0.65 | Clef-flash 0.81 / 0.83, Jev 0.69 / 0.83 |
| Human gate (recall) | 1.00 / 0.33 | Clef 1.00 / 1.00, no false gate |
| Mandatory (accuracy) | 0.64 / 0.87 | Clef-flash 0.90 / 0.48 |
| fan_out (recall) | 0.40 / 0.33 | Clef-flash 0.60 / 0.33 |

The models generalize on kind and gates and the rules do not. Neither wins
mandatory or fan_out, and the current mandatory rule beats every model on
held-out steps. Clef's kind choice was right on 14 of 14 held-out steps
where its confidence was 0.85 or more. The holdout has 3 gates, so the gate
result is small-sample.

Clef and Jev are both wired in as `--decider clef` and `--decider jev`
(a layered `decider` setting). Measured as the shipped deciders, on the
compiled skeleton (end to end, `research/system1-routing/score_deciders.py`),
with Jev's thresholds fitted on dev and frozen before the held-out run:

| Decider | Kind dev / held-out | Gates found dev / held-out | False gates dev / held-out | $ / compile |
| --- | --- | --- | --- | --- |
| rules | 0.64 / 0.49 | 4/4 / 2/3 | 1 / 0 | 0 |
| clef | 0.64 / 0.66 | 4/4 / 3/3 | 2 / 0 | 0.0049 |
| jev | 0.62 / 0.67 | 4/4 / 3/3 | 2 / 1 | 0.0011 |

The method and the fitted thresholds are in
[`docs/design/system1-compile.md`](design/system1-compile.md#--decider-jev).

With the Haiku tie-break on (`--tiebreak haiku`; off by default) for the steps still unsure after
the decider, three trials each, kind dev / held-out: rules 0.63 / 0.68,
clef 0.64 / 0.74, jev 0.67 / 0.72, for about $0.001 a compile. Gates do
not change. The table and method are in
[the design doc](design/system1-compile.md#the-haiku-tie-break---tiebreak-haiku-opt-in), which also has the fidelity bench that kept it off by default.

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
| `gates`, `sig`, `gateJ` | compiled/expected gates, gates with the same signal name, mean Jaccard of the resume payload keys. Each expected gate is compared with a compiled gate, paired among the compiled gates only (`score.match_gates`): the node matcher ignores kind, so it can put a gate's counterpart on a step that is not a gate. The gate pairing has no similarity floor, so the found count in `results.json` (`gates.found`) means paired with some compiled gate, at most the smaller of the two gate counts. A gate at the wrong step still pairs, and shows up as a signal and payload mismatch in its row |
| `sigN`, `gateJN` | the same two on normalized tokens (separators and inflection ignored), so `cs_lead_approved` matches `cs_lead_approval` and `approved_plan` shares `plan` |
| `inR`, `inP` | pipeline input field names, recall and precision |
| `impl` | pure_function nodes implemented / total (the rest are stubs) |
| `tests` | the generated tests, run in the OS sandbox |
| `citeE`, `assum`, `OQ` | citations that are not in the skill, `# rote: assumption:` markers, Open questions entries in compile-report.md. rote renders that report, so `OQ` is distinct assumptions plus stubs. Before that change the agent wrote it, so `OQ` in older results (the baseline) counts the agent's entries |
| `ok` | rote's verification verdict on the deliverable |
| `exit`, `rep`, `turns` | compile exit code, repair rounds, agent turns |
| `cost$`, `inTok`, `outTok` | `total_cost_usd` plus System-1's own model calls (the decider and the tie-break, from `system1/decisions.json`), and the agent's tokens. `inTok` includes cache writes and reads, which are nearly all of it. `results.json` keeps the two parts as `driver_cost_usd` and `decisions_cost_usd` (`null` when a call could not be priced, and then `cost$` leaves it out) |
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

Matching ignores names as far as it can. A compile that calls the
expected `render_welcome_post` `format_announcement` did the same step,
and a name-keyed matcher would charge it a kind error, a mandatory miss
and a spurious mandatory.

Every (expected, compiled) pair gets a similarity: a weighted mean of
the signals that apply to it (`SIGNAL_WEIGHTS` in
`src/rote/bench/score.py`). The signals are:

- the node's topological rank, in- and out-degree, and its neighbors' kinds
- whether its neighbors were matched to the other node's neighbors
- the skill section it cites (`source.section`)
- its input and output field names
- its MCP server and tool
- its description, and only weakly its id

The node's own kind, flags and gate fields are left out, because those
are what the metrics grade. An optimal one-to-one assignment (the
Hungarian algorithm) then maximizes the total similarity, and a pair
under 0.35 is dropped.

Matching runs twice. The second pass adds the matched-neighbors signal
from the first pass's assignment, which settles nodes that look alike
on their own (two fetches at the start) by what they feed. Every metric
reads this one matching: kind, mandatory, fan_out, gates, edges and
waves. `structure.matches` in the JSON lists the pairs and their
similarity.

What changes, measured on the 79 compiles on disk when this landed
(committed snapshots plus every bench run), against 259 hand-labeled
contested matches:

- **Renamed steps are now found.** expense-audit's `find_duplicates` as
  `check_duplicate_submissions`, and customer-onboarding's announce pair.
  The old matcher got 224 of the 259 right, the new one 238.
- **A fully renamed copy of every gold pipeline is recovered 95/95**,
  even with every description blanked. The old matcher recovered 0 with
  descriptions blanked.
- **Splits match the half that sits where the step sat.** The other half
  is an extra node.
- **A moved join does not swap steps.** Degree and neighbor kinds weigh
  what a node feeds 3 to 1 over what feeds it. Compiles often move a
  join (which step takes the duplicate and late flags), but rarely what
  a step feeds, and edgeF1 already charges the moved edge. An earlier
  equal weighting swapped expense-audit's grouping and fanned audit on
  three compiles, and charged a correct fan_out as a miss plus a
  spurious fan (regression fixture:
  `tests/fixtures/bench/expense-audit-joins-moved.yaml`).

Two couplings are left on purpose:

- The structural signals read the same edges edgeF1 and waves grade.
  Structural matching cannot avoid that.
- The `outputs` signal reads only the fields downstream nodes consume,
  never a node's declared output. A gate's declared output is its
  resume payload, which gateJ grades.

`kmix` stays matching-free, as a cross-check.

The expected pipelines are hand-written, so they are one good answer, not
the only one. Compare runs with each other rather than chasing 1.0.

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
  is the tool trace for both. A fixture checks arguments against the
  tool's schema the way a server does: a missing required argument, one
  the tool does not take, or a value of the wrong type is refused with the
  reason, and the case lists it under `refused_calls`. A compile that
  guessed `search_query` for a tool whose argument is `query` gets an
  error, as it would from the real server, not the fixture's catch-all
  answer. A schema without `additionalProperties` is treated as `false`,
  matching FastMCP and other pydantic-validated servers. A server built on
  the TypeScript SDK may ignore extra arguments instead, so against one of
  those a refused unknown argument can overstate the failure; a missing
  required argument fails either way.
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
| `outputs` | stable reference fields the pipeline reproduced. Matched by value anywhere in the pipeline's step outputs and the arguments of its writes (a value it sends on is one it produced; a gate's received answer alone is not), because key names are the compiler's choice. Lists compare as sets, and a list of records per field (the set of `expense_id`s in the review queue). A value that could match by coincidence (an integer under 10, a boolean, a short string, an empty list) also needs a key sharing a word with the field. Gate payloads received are excluded, so a pipeline gets no credit for echoing the approver's answer |
| `outcome` | the run completed when the skill completed, and stopped when the skill stopped |
| `reads` | reads every reference run made, recall, averaged with the share of the pipeline's reads that some reference run also made. A read matches on the data it got (the fixture response that answered it, so a differently worded search for the same data still counts) or, for a live server, on the request it sent. A live read that sent a reference request exactly (server, tool and the arguments as sent, only surrounding whitespace ignored) and got an answer the reference never got is `response_drift`, listed with both answers and not scored: the server reindexed, the pipeline did not change. A read with no answer is still a lost read, and a fixture's answer never drifts. A call is live when its log entry says `"live": true`, which a recording proxy in front of a real server writes and a fixture never does |
| `side_effects` | writes, in order (LCS over the longer sequence) when the reference runs agree on the order, as a set otherwise. A write is identified by its tool and the arguments every reference run sent the same values for: a drafted message body differs in every agent run, so it never decides whether two writes are the same |
| `gates` | key overlap between the case's gate payload and what the compiled gate declares |

`fidelity` is the mean case score. It is a top-level column (`fidel`) in
the table when `--fidelity` is on, with its judge cost under
`fidelity_cost_usd`, apart from the compile cost.

**Reference coverage** (`fcov`, and a line per case in the bench output)
says how much a score can claim: the contract fields that have a stable
value, out of all of them. A case under 50% is flagged `LOW`, with the
reason for each uncovered field: the agent runs disagree on its value,
they agree on the values but put them in different places, or some runs
leave it out. Values compare under layout-normalized paths
(`report.kpi.active_sites` and `report.header.active_sites` are both
`report.active_sites`), so nesting one number differently is not a
disagreement. A leaf collapses only when nothing else in the run lands on
the same path with a different value: `decisions.approved.ids` and
`decisions.rejected.ids` keep their full paths, so two runs that swapped
them stay a disagreement. "Covered" means a field has at least one stable
leaf, so each case also reports its stable leaves out of all leaves. A
case needs at least two successful agent runs: with one, every field
agrees with itself and nothing is stable.

**Fidelity runs code on the host, unsandboxed.** The pipeline side runs
the emitted app with the compile's `extracted/` modules, and when a run
fails, `failure_site` imports the failing node's implementation and calls
it once more to find the line that raised. Both execute agent-written
code with the bench process's permissions, outside the OS sandbox
`rote compile` uses for generated tests. That is acceptable for the
repo's own examples, which are the only skills fidelity runs on. Running
it on an untrusted skill's compile needs that sandbox around both the
pipeline run and the replay first.

### `--backend api` pipelines

An api pipeline calls the vendor directly, so the bench fakes the vendor
instead of the MCP server. A case's `http:` section holds one fixture set
per vendor API version (`google/gmail/v1`), in the shareable format of
`rote.vendor_specs.fixtures`. The bench:

1. reads the backend from `backend.json` beside the compiled IR and emits
   for it;
2. builds a fake of each API from the vendor's own spec
   (`rote.vendor_specs.fake`). The fake rejects what the real API would
   (an unknown operation, a missing credential, an undeclared, mistyped or
   missing required parameter) and answers from the fixtures. With no
   matching fixture it returns the smallest spec-shaped answer: `{}` for an
   empty list, 404 for a file;
3. routes the pipeline process's requests to the fake through a
   `sitecustomize` on its `PYTHONPATH` (`rote.vendor_specs.intercept`:
   httpx, requests, httplib2, urllib and `google.auth.default`). Any other
   host of a vendor in play is blocked, so a fidelity run never reaches a
   real API. Name resolution is guarded too: a client the intercept does
   not patch (`http.client`, `httpcore` or `urllib3` used directly,
   aiohttp) is refused when it resolves a vendor host, `gethostbyname` of
   one is refused, and so is any non-loopback IP literal (the fake listens
   on loopback; judges reach their models by name). This guards
   cooperative clients and is not a sandbox: a raw socket connected to an
   address the code already holds is not seen. Real isolation needs
   egress filtering. Gmail and every other Google API are also answered on
   `www.googleapis.com` at their own paths, as the request check accepts;
4. runs the pipeline with an allowlisted environment
   (`rote.bench.fidelity.bench_environment`): the process essentials, the
   judges' inference credentials, and `DBOS_*` / `ROTE_*` / `LC_*`. A
   vendor credential the user has exported never reaches the run, even
   one read under a name built at run time. The fixture credential the
   bench sets for each name the modules read is the only one there;
5. installs the app's vendor libraries (`google-api-python-client`, ...)
   into a cached target directory, since they are not rote's own
   dependencies;
6. scores the fake's request log like the MCP call log. A fixture's
   `reads: <server>.<tool>[i]` makes an API read and the skill's tool read
   the same read.

Fixture sets can also live in the fixture store
(`rote.vendor_specs.fixture_store`, `ROTE_FIXTURE_DIR`, default
`~/.local/share/rote/fixtures`), one JSON file per set at
`<vendor>/<api>/<version>/<name>.json`, which is the layout a shared store
will hold. `rote bench --examples examples --skills ops-report
--save-fixtures` validates every case's sets against the live specs and
saves them as `<example>.<case>`, stamped with the spec revision they fit.
A case uses a stored set with `http: {google/gmail/v1: {from_store: <name>}}`,
and loading one re-validates it against the current spec.

### What fidelity does and does not measure

Read the number with these limits in mind:

- **It measures agreement with the agent, not with the gold.** The
  reference is the skill run as an agent. Where the agent reads the skill
  differently from the hand-written pipeline, the gold loses points too
  (see below). Where the agent is inconsistent, that field is not scored.
- **Values match anywhere.** A reference value counts as reproduced when
  it appears in any step output. A field that passes data through
  unchanged (an id, a period) is credited as soon as the step that
  fetched it runs, whether or not anything downstream used it. The
  converse also holds: a value the pipeline only holds as a filtered
  subset of a larger list is not found (see `blocked_ids` below).
- **The contract is summaries.** Counts, flagged ids, totals. The
  individual dwell, sorter and container rows behind a summary are not
  compared, so a pipeline can get a row-level detail wrong and keep a
  correct total.
- **An api pipeline's writes are not scored.** The fake never reads a
  request body, and a write is keyed by its operation and its query and
  path parameters, so it can never equal the skill's MCP write arguments.
  An api pipeline that writes scores 0 on side effects whether the write
  is right or wrong. ops-report makes no writes, so its api scores are
  unaffected; a skill that writes needs body-level write fixtures first.
- **expense-audit exercises no reads or side effects.** It uses no MCP,
  and the report file is not compared. Its score is outputs, outcome and
  gate fit only.
- **The fixture shapes are assumptions.** The ops-report fixtures answer
  Gmail `search_threads` with a bare JSON list of threads and Drive
  `read_file_content` with plain CSV text. The skill names the tools but
  not their result shapes, and so do the fixtures' authors: neither shape
  was recorded from the real servers. Eight compiles in the October sweep
  read `.threads` and `.content` from those results and fail here. That
  failure is real only if the real servers answer the way the fixtures
  do. Record real responses (a `rote baseline` run against the live
  servers gives them) before treating it as a compile defect.
- **An input the skill does not name is a contract the compiler invents.**
  expense-audit's skill describes its input ("one row per expense line,
  plus the reporting period") without naming a field. The cases pass the
  bundled `input.json` (`period`, `submissions`, rows keyed `id`), and
  every compile in the sweep named those differently, so none ran. That
  is a real failure (the pipeline cannot take the export it is given),
  but it says nothing about the rest of the pipeline until compile is
  grounded in a sample input.

### Checking that the metric measures behavior

The gold pipeline (expense-audit's hand-written `expected/`) and two
broken copies, scored with `--fidelity-of`:

| Pipeline | fidelity | base | second_batch | unknown_currency |
| --- | --- | --- | --- | --- |
| gold | 0.96 | 1.00 | 0.98 | 0.89 |
| mandatory duplicate check dropped | 0.87 | 0.94 | 0.88 | 0.78 |
| `duplicate_ids` and `late_ids` inputs swapped | 0.66 | 0.58 | 0.52 | 0.89 |
| the pre-#144 baseline compile | 0.17 | 0.00 | 0.00 | 0.50 |

Each break costs what it should: the dropped check loses the duplicate
id and the review-queue ids, and the swap crashes the report (so it
loses `outcome` too) except on the case that stops before the checks
matter. The baseline compile renamed the pipeline input (`lines`,
`fx_rates` instead of `period`, `submissions`), so it cannot run on the
skill's own input at all. The structural score had the input recall at
0.5 and nothing else alarming.

The gold misses `blocked_ids` on second_batch although it computes the
same four ids the agent reports: it never holds them as a list of their
own, only as the review-queue rows whose severity is `block`. Value
matching finds a list or a column, not a filtered subset of one, so a
field the agent derives by filtering can be missed even when the pipeline
got it right. The base case shows the agent's own variance: its three
runs disagree on `reimbursable_usd` ($2,544.73, $2,990.98, $2,544.73),
so that field is unstable there and not scored.

Costs: the references cost $1.50 (ops-report, 6 runs) and $4.48
(expense-audit, 9 runs, recorded a second time after the reference prompt
began asking for `null` on fields a stopped run never computed: the first
recording, $4.51, had the stop case at 4 of 13 stable fields because
agents wrote 0, `[]` or nothing for the same uncomputed field. It is at
13 of 13 now). Per-compile fidelity on these two
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
