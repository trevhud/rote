# rote

**Compile AI agent skills into cheap, fast, deterministic pipelines that
run without an LLM in the loop.**

[![CI](https://github.com/trevhud/rote/actions/workflows/ci.yml/badge.svg)](https://github.com/trevhud/rote/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/rote-cli.svg)](https://pypi.org/project/rote-cli/)
[![Downloads](https://img.shields.io/pypi/dm/rote-cli.svg)](https://pypi.org/project/rote-cli/)
[![Python versions](https://img.shields.io/pypi/pyversions/rote-cli.svg)](https://pypi.org/project/rote-cli/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

**Your Claude skill works. Running it a thousand times does not.**

`rote` is an open source CLI that compiles a proven Claude skill or
agent skill (a `SKILL.md` plus `references/`) into a typed,
deterministic pipeline. It moves the fixed logic and tool orchestration
into reviewable code, and calls a model only for the steps that
genuinely need judgment. A 10 to 20 minute agent loop becomes a
background workflow that costs a fraction of the tokens and can be
regression tested.

```sh
pip install rote-cli    # or zero-install: uvx --from rote-cli rote ...

# `rote compile` runs an LLM agent, so it needs a driver: Claude Code
# (`claude`) or Codex (`codex`) installed and authed, or ANTHROPIC_API_KEY
# for the in-process `api` driver. The BDR run below takes ~13 min and
# ~$0.70 with Sonnet. (`rote emit` needs no LLM; see below.)

# Default target is DBOS: durable execution as a plain Python library,
# no orchestrator to run, SQLite for dev / Postgres for prod:
rote compile ./examples/bdr-outreach/skill --out ./compiled/

# Or pick another runtime (see the table below):
rote compile ./examples/bdr-outreach/skill --runtime temporal   --out ./compiled/
rote compile ./examples/bdr-outreach/skill --runtime cloudflare --out ./compiled/
```

The name comes from *rote learning*: doing something so many times, so
reliably, that it becomes mechanical. That's what compilation does to a
skill.

---

## Why

Agent skills work, but repeating them in production is expensive, slow,
and non-deterministic. Token cost is what bites first: every run
re-reads the skill text, the tool schemas, and a growing transcript to
re-derive a procedure the author already wrote down. The two production
skills adapted into `examples/` averaged ~0.9M and ~1.6M cache-read
tokens *per run* before compilation, for work that is mostly arithmetic
and fixed API calls. Latency is next (a 10 to 20 minute agent loop is
unacceptable as a background job), then determinism: a "MANDATORY"
check enforced only by prose can be silently skipped, and there's no
way to regression-test a behavior the LLM has to remember.

The fix is to separate the parts of a skill that are *actually* fuzzy
from the deterministic procedures wearing fuzzy clothing. Move the
deterministic parts into code, keep the LLM only where the input is
genuinely unbounded (parsing, classifying, drafting), and wrap the whole
thing in a durable execution engine with explicit human-in-the-loop
gates. That compilation step is what `rote` automates.

| | Run the skill as an agent, every time | Compile it with `rote` |
| --- | --- | --- |
| Tokens per run | Full agent loop | Only the judgment steps |
| Latency | 10 to 20 minutes | Background, seconds to minutes |
| Reproducibility | Prose `MANDATORY` can be silently skipped | Deterministic nodes always run |
| Testing | No per-step regression tests | Typed nodes, per-step tests, eval seeds |
| Failure recovery | Restart the loop | Durable retries and resume |
| Human approval | Ad hoc | Explicit HITL gates that suspend and resume |

There's third-party data for what this buys. ["Compiled AI: Deterministic
Code Generation for LLM-Based Workflow Automation"](https://arxiv.org/abs/2604.05150)
(Trooskens et al., Apr 2026) measured compiling LLM workflows into
deterministic code: **57× fewer tokens** at 1,000 transactions, **450×
lower** median latency, **100% reproducibility** (vs. 95% for direct
inference at temperature 0), and ~**40× lower TCO** at a million
transactions a month. The multiples grow with volume. Once a workflow
is proven, every run through an agent loop pays LLM prices for work code
does for free.

A distinction worth being precise about: durable-execution vendors make
fuzzy agents *durable* (wrap the loop in retries and state so it survives
crashes, still fuzzy inside). `rote` *removes* the fuzzy loop. The two
compose: Temporal, Cloudflare Workflows, and the rest are `rote`'s
compile targets, not its rivals.

**When not to use `rote`:** exploratory and one-off work should stay an
agent loop. Flexibility is the whole point there, and there's nothing
proven to compile yet. `rote` is for the skill you've run twenty times
and want to run a thousand more, unattended.

---

## How it works

One `rote compile` run does the whole thing. An LLM agent (itself
defined as a skill) reads the source skill, applies a structured
compilation rubric, and emits a runtime-agnostic intermediate
representation (`pipeline.yaml`), extracted Python modules for the
deterministic parts, typed signature stubs for the LLM-judge parts, and
runnable code for the durable execution engine of your choice.

`rote` is a three-layer system; each layer has one job and contracts on
a small interface.

```
   SKILL.md + references/          Source skill bundle (untouched)
             │  rote compile
             ▼
   compiler agent                 An LLM agent (Claude / Codex /
   (pluggable driver)              Anthropic SDK) runs the rote-compile
             │                     skill against the source bundle.
             │  filesystem contract: work_dir/pipeline.yaml
             ▼                     + extracted/ + signatures/
   Pipeline IR (pipeline.yaml)     Pydantic-validated DAG of typed
             │                     nodes. Five node kinds. Runtime-agnostic.
             │  rote.adapters.<runtime>
             ▼
   emitted runtime code            Native code for the target durable
                                   execution engine.
```

1. **The compiler agent** (`skills/rote-compile/`): a regular
   Anthropic Skill (`SKILL.md` + four reference files). This is the
   *brain*; it runs inside any Skills-compatible surface, and you don't
   need `rote` to use it.
2. **The IR** (`src/rote/ir.py`): Pydantic models for the five node
   kinds plus edges, retries, HITL gates, and metadata. The IR is the
   source of truth; everything downstream is template substitution.
3. **Runtime adapters** (`src/rote/adapters/`): pluggable modules that
   consume an IR and emit runnable code for one engine.

The compiler's job ends when it has produced a valid `pipeline.yaml`.
Code emission is *deterministic Python*, never agent-driven, so the
same IR always produces byte-identical output.

---

## Quickstart

### From Claude Code (recommended)

`rote` ships as a Claude Code plugin, so you can compile a skill without
touching Python tooling:

```
/plugin marketplace add trevhud/rote
/plugin install rote@rote
```

Then say "compile this skill" (or run `/rote:compile`). It confirms the
source directory, asks which runtime you want, runs the CLI via
[uv](https://docs.astral.sh/uv/) in the background, and reports the
emitted pipeline. A second skill, `/rote:serve`, wires compiled
pipelines up as MCP tools so Claude can trigger the deployed workflows
(see [docs/mcp-trigger.md](docs/mcp-trigger.md)).

Prefer a terminal? The same thing is one `uvx` command:

```sh
uvx --from rote-cli rote compile ./my-skill --runtime dbos --out ./compiled
```

**Hosted platform:** [roteskills.com](https://roteskills.com) is the
project site (concepts, benchmarks methodology, worked examples).
[app.roteskills.com](https://app.roteskills.com) is Rote Cloud, a
managed path for teams that would rather not operate a runtime: run
`rote login` and `rote compile` then runs server-side, streams progress
back, auto-deploys, and downloads the artifacts locally. Everything in
this README still works logged out.

> **Naming note:** the `rote` package on PyPI is an unrelated
> memoization library that also installs `import rote`, so the two can't
> share an environment. This project's distribution is `rote-cli` while
> the CLI command and import name stay `rote`, hence
> `uvx --from rote-cli rote ...`. See [docs/releasing.md](docs/releasing.md).

### Run on the bundled example

The repo includes a real BDR outreach skill (lead generation, contact
vetting, CRM upload, mandatory exclusion checks, email personalization,
manual enrollment handoff) in `examples/bdr-outreach/skill/`:

```sh
rote compile examples/bdr-outreach/skill --out /tmp/bdr-compiled
```

On that skill the compiler produces a 22-node IR that's **78.9%
codifiable** (15 of 19 non-gate nodes), extracts 5 Python modules and 2
typed judge signatures, and flags 4 mandatory nodes and 3 HITL gates,
all in ~13 minutes for ~$0.70 (Sonnet via Claude Code). Along the way it
independently lifts the three MANDATORY exclusion checks out of prose,
pulls four batch-size constants out of prompt text, and models a
parallel entry path the hand-written baseline missed.

`rote` auto-detects a driver in the order `claude` → `codex` → `api`;
override with `--agent`. The output directory splits into `compiled/`
(the agent's `pipeline.yaml`, `extracted/`, `signatures/`, eval seeds,
and a `compile-report.md`) and `runtime/<runtime>/` (the adapter's
emitted code + a README on how to run, signal gates, and deploy).
`compiled/compile-metrics.json` preserves final token counts, session ID,
and the driver's cost estimate before emission starts, so an adapter
failure does not lose the paid run's metrics. Missing final output stays
null with `usage_complete: false`; unchanged updates retain prior metrics.

Local `api` and `openai-api` compilation automatically discovers your
registered, authenticated MCP servers and exposes their read-only tools.
Pass `--no-mcp` to skip that discovery and compile without live MCP calls:

```sh
rote compile examples/bdr-outreach/skill --local --agent api --no-mcp --out /tmp/bdr-compiled
```

This leaves the emitted pipeline's MCP bindings intact. The flag also works
with `rote analyze` and with an API driver selected through config or
auto-detection. It rejects cloud compilation, subprocess drivers, and
`--baseline`, whose raw skill run uses its own MCP tools. API drivers bill
through your API credentials.

### Other commands

- **`rote emit <pipeline.yaml> --out <dir>`**: run just the adapter step
  on an existing IR. LLM-free, so no cost and no driver needed, which
  makes it the cheap inner loop while iterating on adapters or IR
  shapes. Re-emitting is safe: a `.rote-manifest.json`
  tracks what `rote` wrote, and files you've edited are left untouched
  (the fresh version lands as `<name>.new`).
- **`rote compile --update`**: re-compile incrementally when the skill
  changes. `rote` diffs the skill against the previous run's
  `provenance.json` and re-derives only the nodes whose source sections
  changed; unchanged nodes keep their ids (so in-flight durable workflows
  aren't orphaned) and implemented stubs are kept. No change → no agent run.
- **`rote run <path>`**: one-off local execution of either side. A
  skill directory runs as an agent via `claude -p` (your registered MCP
  servers injected, read-only tool gate unless `--allow-writes`); an
  emitted runtime directory, or a `compile --out` directory, runs
  the pipeline itself on any of the six runtimes (`python`/`dbos`/`temporal`
  in-process or on a managed local dev server, `cloudflare` under
  `wrangler dev`, `inngest` against a managed `inngest-cli dev`,
  `dbos-ts` against your Postgres or a throwaway Docker one). HITL
  gate payloads via `--signal name='{...}'` or an interactive prompt.
  Runtimes that bundle a dev UI surface it: temporal runs print a live
  Temporal Web UI URL and inngest runs print the dev-server dashboard,
  both live for the duration of the run. Output JSON on stdout, status
  on stderr, so it pipes.
- **`rote deploy <path>`**: push an emitted pipeline where it runs:
  `cloudflare` wraps `npx wrangler deploy` (with `--dry-run`), `dbos` /
  `dbos-ts` wrap `npx dbos-cloud app deploy`, and the vendor CLI owns auth
  and output; rote adds detection and preflights (including surfacing
  *which* account your wrangler session belongs to before uploading).
  Runtimes with no push model (temporal, inngest, python) print honest
  hosting guidance with doc links instead of a fake action.
  `--target rote-cloud` bundles a cloudflare-emitted app (esbuild via
  npx) and uploads it to a hosted rote-cloud instance. With a stored
  `rote login`, no flags or env vars are needed (`--url`/`--token` and
  `$ROTE_CLOUD_URL`/`$ROTE_CLOUD_TOKEN` still override).
- **`rote login`**: connect the CLI to a rote-cloud account via the
  OAuth device flow: your browser opens with a one-time code pre-filled
  (over SSH, `--device` prints the code + URL instead), you click
  Approve, and the CLI stores a tenant API key at
  `~/.local/share/rote/cloud.json` (mode 0600). Once logged in,
  `rote compile` runs **on rote cloud by default**: the skill bundle
  syncs up (sha-diffed, so unchanged files don't re-upload), the
  platform runs the compilation server-side, live progress streams back
  through the same renderer as a local run, the result auto-deploys,
  and the artifacts download into your `--out` directory in the exact
  local layout. `--local` keeps the compilation on your machine (then
  the cloudflare-emit + auto-deploy flow applies), `--no-deploy` or a
  config opt-out (`runtime:` pinned to a local target, or
  `deploy: none`) keeps everything local; `--cloud` forces the server
  even where config says otherwise. Logged out, everything works
  locally exactly as before. `rote whoami` shows the account (verified
  live); `rote logout` revokes the key server-side and clears the store.
- **`rote init`**: one-time interactive onboarding: pick where
  compiled pipelines run (rote cloud, with login offered inline, or
  a local runtime, with a one-line pitch for each), which compiler
  driver does the work (availability probed live), and optionally a
  model. Answers are saved to `~/.config/rote/config.yaml`
  (`--project` writes a `./rote.yaml` that overrides it per-repo) and
  every later command reads them. It's the only interactive command
  besides login; CI never hits a prompt.
- **`rote config`**: print every configurable default with its
  effective value *and the layer that set it*. Resolution everywhere is
  `flag > ROTE_* env (ROTE_RUNTIME, ROTE_DEPLOY, ROTE_AGENT,
  ROTE_MODEL) > project rote.yaml > user config > built-in`. Config
  files are strict: a typo'd key or value is a loud error, never a
  silent fallback. `--json` for automation.
- **`rote eval <compiled>`**: render the before/after scorecard (wall
  clock, cost across the current model lineup at live prices, and how
  much of the run is still LLM-decided). `rote compile` writes this to
  `compiled/scorecard.md` automatically. Add `--run` to *measure*
  instead of estimate: it executes both sides for real and appends
  measured cost, turns, and output agreement across trials.
- **Per-node inference**: emitted judges read `ROTE_MODEL_<ID>` and
  `ROTE_BASE_URL_<ID>` at runtime, so you can swap the model or point at
  any OpenAI-compatible endpoint (Ollama, vLLM, a gateway) without
  re-emitting.

---

## The five node kinds

Every step in a compiled pipeline is exactly one of five kinds. Full
guidance:
[`references/node-kinds.md`](skills/rote-compile/references/node-kinds.md).

| Kind | What it is | Where the LLM lives |
| --- | --- | --- |
| `pure_function` | Fixed logic, deterministic I/O | Not involved |
| `external_call` | Vendor API call with fixed semantics + retries | Not involved |
| `llm_judge` | Fuzzy classification against a rubric, typed I/O | Typed signature (DSPy/BAML in Python; Zod + vendor SDK in TS), from the IR's runtime-agnostic `signature_spec` |
| `agent_loop` | Genuinely exploratory tool use | Bounded agent loop |
| `hitl_gate` | Explicit human approval, suspend until signal | Durable suspend/resume |

The guiding rule: **keep the LLM at points where the input is unbounded
or ambiguous, and codify everything else.** When a step could go either
way, prefer the more deterministic kind.

---

## Runtimes

Pick with `--runtime`; the same IR drives all of them. Under
`--backend api`, none of the emitted code references MCP: the
crystallization step replaces tool calls with direct vendor API calls.
Under the default `--backend mcp`, tool-using nodes emit a working MCP
client call (with durable park-on-auth on every MCP-capable runtime);
see [`docs/mcp-client.md`](docs/mcp-client.md).

| Runtime | `--runtime` | Language | Shape | Notes |
| --- | --- | --- | --- | --- |
| **DBOS** (default) | `dbos` | Python | `main.py` with `@DBOS.workflow` + `@DBOS.step` per node | No orchestrator to deploy; SQLite (dev) / Postgres (prod) |
| Temporal | `temporal` | Python | `workflow.py` + `activities.py` | Signal handlers for HITL gates |
| Plain Python | `python` | Python | single `main.py` script | Max legibility, stdlib only; refuses HITL-gate pipelines |
| Cloudflare Workflows | `cloudflare` | TypeScript | `WorkflowEntrypoint` + `wrangler.jsonc` | `wrangler deploy`-ready |
| DBOS (TypeScript) | `dbos-ts` | TypeScript | `src/main.ts` (DBOS Transact) | Zero-orchestrator; Postgres-only |
| Inngest | `inngest` | TypeScript | one `inngest.createFunction` | Mounts into an existing Node/Next.js app; retries are function-level |

---

## Drivers

`rote` ships three interchangeable compiler drivers. Pick whichever
matches your auth. The same `pipeline.yaml` comes out either way.

| Driver | Backend | Auth | Install |
| --- | --- | --- | --- |
| `claude` (default) | `claude -p` subprocess | Claude Max/Pro OAuth or `CLAUDE_CODE_OAUTH_TOKEN` | Install Claude Code separately |
| `codex` | `codex exec` subprocess | ChatGPT Plus/Pro OAuth | Install Codex CLI separately |
| `api` | `anthropic` Python SDK | `ANTHROPIC_API_KEY` | `pip install 'rote-cli[api]'` |

The `claude` driver scrubs `ANTHROPIC_API_KEY` from the subprocess so a
subscription login wins, and limits the agent to read/write/glob/grep
tools. The default model is **Sonnet** rather than Opus, because the
task is structured-rubric-following, not deep reasoning; Sonnet brings
per-run cost from ~$3.50 to ~$0.70. Override with `--model` for skills
where Opus earns its cost. Full design record, including the auth gotcha:
[docs/agent-runtime.md](docs/agent-runtime.md).

`rote` explicitly **does not** depend on `claude-agent-sdk`: Anthropic's
ToS forbids third-party agents built on the Agent SDK from using
claude.ai login credentials without approval, which would defeat the
subscription path.

---

## How it differs from other tools

- **vs. raw durable engines (Temporal / Cloudflare / Inngest / Restate):**
  they give you the workflow *runtime*; they don't help you decide *what
  should be a workflow*. `rote` is the missing step that turns a working
  skill into something worth running on one.
- **vs. LangGraph:** LangGraph is an excellent state machine, but its
  graph is hand-built. `rote` produces a graph *from prose*, classifies
  nodes by determinism, and pushes work out of the agent loop wherever
  the data supports it.
- **vs. using Skills directly:** Skills run great interactively. `rote`
  is what you reach for when a skill becomes business-critical and needs
  to run unattended with hard reliability guarantees and per-step
  regression tests.

---

## Status

`rote` is **pre-1.0**. The end-to-end flow works on the BDR example. The
fast suite (`pytest tests/`) makes no real API calls and is what CI runs
on every push, alongside a Python e2e (DBOS over SQLite + the MCP server
over real stdio). Each adapter also has a `slow`-marked e2e that runs its
emitted code against the real runtime (Temporal's time-skipping server,
the TypeScript targets via `tsc --noEmit` and live dev servers, the
plain-Python subprocess); those need a Node toolchain / Docker, so they
run locally with `pytest tests/ -m slow`, not in CI.

Known gaps: the extracted modules are `NotImplementedError` stubs
you fill in with real API-client code, and a Restate adapter is planned.
Published on
PyPI as [`rote-cli`](https://pypi.org/project/rote-cli/) via tag-driven
Trusted Publishing ([docs/releasing.md](docs/releasing.md)).

On the numbers: static scorecard estimates, observed production-agent
baselines, and independent research are three different kinds of
evidence, and mixing them produces marketing rather than benchmarks.
They're kept separate, with the assumptions written out, at
[roteskills.com/benchmarks](https://roteskills.com/benchmarks). To
measure your own workflow instead of reading someone else's, use
`rote eval --run`.

---

## Repository layout

```
rote/
├── docs/                  agent-runtime · mcp-client · mcp-trigger · releasing
├── skills/rote-compile/  the compiler agent (SKILL.md + 4 reference files)
├── src/rote/
│   ├── cli.py             rote compile / emit / eval / serve
│   ├── ir.py              Pydantic IR models + load_pipeline
│   ├── compiler/         orchestrator + drivers/ (claude · codex · anthropic_api)
│   └── adapters/          dbos · temporal · python · cloudflare · dbos_ts · inngest
│                          (+ _common / _py_common / _ts_common emit helpers)
├── examples/
│   ├── bdr-outreach/      canonical: all 5 node kinds · IR baseline · run snapshots
│   ├── ops-report/        100% roteness: zero LLM nodes + a HITL gate
│   ├── deal-monitor/      data-heavy: parallel waves · fan-out judges · template render
│   └── invoice-push/      agent-loop archetype: bounded browser loop · turn-dominated cost
└── tests/                 fast + slow suites (pytest -m slow)
```

---

## Documentation

- [`AGENTS.md`](AGENTS.md): operating manual for a coding agent *driving*
  `rote` as an installed tool (invocation contract, the slow/costs-money
  `compile` flow, auth, failure recovery, the stub-filling job, `--json`)
- [`docs/agent-runtime.md`](docs/agent-runtime.md): design record for the
  driver abstraction (the `claude -p` env gotcha; the non-use of
  `claude-agent-sdk`)
- [`docs/mcp-client.md`](docs/mcp-client.md): the OAuth MCP client
  emitted code uses under `--backend mcp`: endpoint/credential
  resolution and durable park-on-auth across every MCP-capable runtime
- [`docs/mcp-trigger.md`](docs/mcp-trigger.md): `rote register` +
  `rote serve`: compiled pipelines as MCP tools (FastMCP 3.x)
- [`docs/releasing.md`](docs/releasing.md): tag-driven PyPI Trusted
  Publishing
- [`skills/rote-compile/`](skills/rote-compile/): the compiler's
  `SKILL.md` and its four rubric files (node kinds, crystallization
  heuristics, IR schema, LLM-judge extraction)
- [`examples/bdr-outreach/`](examples/bdr-outreach/): the canonical
  skill, its ground-truth IR, and snapshotted real compiler runs
- [`examples/ops-report/`](examples/ops-report/): the 100%-roteness
  archetype: every step deterministic, one durable HITL gate, zero LLM
  nodes after compilation
- [`examples/deal-monitor/`](examples/deal-monitor/): the data-heavy
  archetype: parallel entry waves, fan-out judges, and a template render
  replacing per-run LLM-generated HTML
- [`examples/invoice-push/`](examples/invoice-push/): the `agent_loop`
  archetype: a bounded browser-automation loop stays one agent node
  while the date math, filtering, and reporting around it compile to
  code, plus the measured runs that forced the loop-aware cost model

---

## Roadmap

In rough priority order:

1. **Pre-filter as a `pure_function` node**: today hard thresholds are
   lifted into a judge's `forward()`, which works for the Python
   runtimes but not the TypeScript ones; a separate node makes the
   short-circuit uniform across all six.
2. **The compiler compiling itself**: `rote-compile` is a SKILL.md;
   pointing `rote compile` at it should crystallize its rubric-grade
   pieces and leave only the genuinely fuzzy judgments in the loop.
3. **More example skills**: four archetypes are covered today (BDR
   outreach, 100%-roteness, data-heavy, agent-loop). Research-heavy,
   retrieval-heavy, and code-review skills each stress the IR in ways
   none of those four do.

---

## FAQ

### What is rote?

`rote` is an open source CLI, Apache-2.0 licensed and published as
[`rote-cli`](https://pypi.org/project/rote-cli/) on PyPI, that compiles
a proven AI agent skill into a typed, deterministic pipeline. It reads
an Anthropic-style `SKILL.md`, classifies each step, moves fixed logic
and tool orchestration into reviewable code, and calls a model only for
the steps that genuinely require judgment.

### How does rote reduce AI agent token costs?

It removes model calls rather than making them cheaper. A repeating
agent spends tokens re-reading instructions, tool schemas, and history
to re-derive a procedure it already established. `rote` compiles that
procedure into code, so a repeated run pays only for the steps still
classified as needing judgment.

### When should I compile a skill instead of leaving it as an agent?

Keep one-off exploration in an agent, which is what agents are good at.
Compile a skill once the procedure is proven, repeats often, and needs
lower cost, faster execution, regression tests, explicit approvals, or
reliable retries.

### Does rote replace my agent framework or MCP?

No. A compiled workflow can still call authenticated MCP servers and
retain bounded agent loops. `rote` decides which parts of a process
should stop being inference; your runtime and your integrations stay
where they are.

### Where does the compiled workflow run?

Anywhere you already run durable work. `rote` emits DBOS, Temporal,
Cloudflare Workflows, plain Python, DBOS TypeScript, and Inngest.
[Rote Cloud](https://app.roteskills.com) is an optional managed path for
teams that would rather not operate the runtime themselves.

---

## Contributing

The most useful contribution right now is to **run `rote compile` on a
real skill of your own and report what happens**. The rubric was
designed against one skill and needs more. Adding a runtime adapter or a
compiler driver, or improving the rubric, are all good next steps. See
[CONTRIBUTING.md](CONTRIBUTING.md) for dev setup, the test layout, and
the adapter/driver how-tos.

## License

Apache-2.0. See [LICENSE](LICENSE).
