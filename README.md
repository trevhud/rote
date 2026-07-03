# rote

**Graduate fuzzy AI skills into deterministic, reliable workflows.**

[![CI](https://github.com/trevhud/rote/actions/workflows/ci.yml/badge.svg)](https://github.com/trevhud/rote/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/rote-cli.svg)](https://pypi.org/project/rote-cli/)
[![Python versions](https://img.shields.io/pypi/pyversions/rote-cli.svg)](https://pypi.org/project/rote-cli/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

`rote` is a CLI that takes an Anthropic-style Skill (a `SKILL.md` plus
`references/`) and turns it into a runnable background pipeline in one
shot. An LLM agent (itself defined as a skill) reads the source skill,
applies a structured graduation rubric, and emits a runtime-agnostic
intermediate representation (`pipeline.yaml`), extracted Python modules
for the deterministic parts, typed signature stubs for the LLM-judge
parts, and runnable code for the durable execution engine of your
choice.

```sh
pip install rote-cli    # or zero-install: uvx --from rote-cli rote ...

# `rote graduate` runs an LLM agent, so it needs a driver: Claude Code
# (`claude`) or Codex (`codex`) installed and authed, or ANTHROPIC_API_KEY
# for the in-process `api` driver. The BDR run below takes ~13 min and
# ~$0.70 with Sonnet. (`rote emit` needs no LLM — see below.)

# Default target is DBOS — durable execution as a plain Python library,
# no orchestrator to run, SQLite for dev / Postgres for prod:
rote graduate ./examples/bdr-outreach/skill --out ./graduated/

# Or pick another runtime (see the table below):
rote graduate ./examples/bdr-outreach/skill --runtime temporal   --out ./graduated/
rote graduate ./examples/bdr-outreach/skill --runtime cloudflare --out ./graduated/
```

The name comes from *rote learning* — doing something so many times, so
reliably, that it becomes mechanical. That's what graduation does to a
skill: a fuzzy 10–20 minute agent loop becomes a deterministic pipeline
that runs in the background, costs a fraction of the tokens, and can be
regression-tested.

---

## Why

Fuzzy AI skills work, but in production they're slow (a 10–20 minute
agent loop is unacceptable as a background job), expensive (multi-agent
loops use ~15× the tokens of a single chat, mostly re-deriving
procedures the author already wrote down), and non-deterministic (a
"MANDATORY" check enforced only by prose can be silently skipped, and
there's no way to regression-test a behavior the LLM has to remember).

The fix is to separate the parts of a skill that are *actually* fuzzy
from the deterministic procedures wearing fuzzy clothing. Move the
deterministic parts into code, keep the LLM only where the input is
genuinely unbounded (parsing, classifying, drafting), and wrap the whole
thing in a durable execution engine with explicit human-in-the-loop
gates. That graduation step is what `rote` automates.

There's third-party data for what this buys. ["Compiled AI: Deterministic
Code Generation for LLM-Based Workflow Automation"](https://arxiv.org/abs/2604.05150)
(Trooskens et al., Apr 2026) measured compiling LLM workflows into
deterministic code: **57× fewer tokens** at 1,000 transactions, **450×
lower** median latency, **100% reproducibility** (vs. 95% for direct
inference at temperature 0), and ~**40× lower TCO** at a million
transactions a month. The multiples grow with volume — once a workflow
is proven, every run through an agent loop pays LLM prices for work code
does for free.

A distinction worth being precise about: durable-execution vendors make
fuzzy agents *durable* (wrap the loop in retries and state so it survives
crashes — still fuzzy inside). `rote` *removes* the fuzzy loop. The two
compose: Temporal, Cloudflare Workflows, and the rest are `rote`'s
compile targets, not its rivals.

**When not to use `rote`:** exploratory and one-off work should stay an
agent loop — flexibility is the whole point there, and there's nothing
proven to compile yet. `rote` is for the skill you've run twenty times
and want to run a thousand more, unattended.

---

## How it works

`rote` is a three-layer system; each layer has one job and contracts on
a small interface.

```
   SKILL.md + references/          Source skill bundle (untouched)
             │  rote graduate
             ▼
   graduator agent                 An LLM agent (Claude / Codex /
   (pluggable driver)              Anthropic SDK) runs the rote-graduate
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

1. **The graduator agent** (`skills/rote-graduate/`) — a regular
   Anthropic Skill (`SKILL.md` + four reference files). This is the
   *brain*; it runs inside any Skills-compatible surface, and you don't
   need `rote` to use it.
2. **The IR** (`src/rote/ir.py`) — Pydantic models for the five node
   kinds plus edges, retries, HITL gates, and metadata. The IR is the
   source of truth; everything downstream is template substitution.
3. **Runtime adapters** (`src/rote/adapters/`) — pluggable modules that
   consume an IR and emit runnable code for one engine.

The graduator's job ends when it has produced a valid `pipeline.yaml`.
Code emission is *deterministic Python* — never agent-driven — so the
same IR always produces byte-identical output.

---

## Quickstart

### From Claude Code (recommended)

`rote` ships as a Claude Code plugin, so you can graduate a skill without
touching Python tooling:

```
/plugin marketplace add trevhud/rote
/plugin install rote@rote
```

Then say "graduate this skill" (or run `/rote:graduate`). It confirms the
source directory, asks which runtime you want, runs the CLI via
[uv](https://docs.astral.sh/uv/) in the background, and reports the
emitted pipeline. A second skill, `/rote:serve`, wires graduated
pipelines up as MCP tools so Claude can trigger the deployed workflows
(see [docs/mcp-trigger.md](docs/mcp-trigger.md)).

Prefer a terminal? The same thing is one `uvx` command:

```sh
uvx --from rote-cli rote graduate ./my-skill --runtime dbos --out ./graduated
```

> **Naming note:** the `rote` package on PyPI is an unrelated
> memoization library that also installs `import rote`, so the two can't
> share an environment. This project's distribution is `rote-cli` while
> the CLI command and import name stay `rote` — hence
> `uvx --from rote-cli rote ...`. See [docs/releasing.md](docs/releasing.md).

### Run on the bundled example

The repo includes a real BDR outreach skill (lead generation, contact
vetting, CRM upload, mandatory exclusion checks, email personalization,
manual enrollment handoff) in `examples/bdr-outreach/skill/`:

```sh
rote graduate examples/bdr-outreach/skill --out /tmp/bdr-graduated
```

On that skill the graduator produces a 22-node IR that's **78.9%
codifiable** (15 of 19 non-gate nodes), extracts 5 Python modules and 2
typed judge signatures, and flags 4 mandatory nodes and 3 HITL gates —
in ~13 minutes for ~$0.70 (Sonnet via Claude Code). Along the way it
independently lifts the three MANDATORY exclusion checks out of prose,
pulls four batch-size constants out of prompt text, and models a
parallel entry path the hand-written baseline missed.

`rote` auto-detects a driver in the order `claude` → `codex` → `api`;
override with `--agent`. The output directory splits into `graduated/`
(the agent's `pipeline.yaml`, `extracted/`, `signatures/`, eval seeds,
and a `graduation-report.md`) and `runtime/<runtime>/` (the adapter's
emitted code + a README on how to run, signal gates, and deploy).

### Other commands

- **`rote emit <pipeline.yaml>`** — run just the adapter step on an
  existing IR (no LLM, no cost). The cheap inner loop while iterating on
  adapters or IR shapes. Re-emitting is safe: a `.rote-manifest.json`
  tracks what `rote` wrote, and files you've edited are left untouched
  (the fresh version lands as `<name>.new`).
- **`rote graduate --update`** — re-graduate incrementally when the skill
  changes. `rote` diffs the skill against the previous run's
  `provenance.json` and re-derives only the nodes whose source sections
  changed; unchanged nodes keep their ids (so in-flight durable workflows
  aren't orphaned) and implemented stubs are kept. No change → no agent run.
- **`rote eval <graduated>`** — render the before/after scorecard (wall
  clock, cost across the current model lineup at live prices, and how
  much of the run is still LLM-decided). `rote graduate` writes this to
  `graduated/scorecard.md` automatically. Add `--run` to *measure*
  instead of estimate: it executes both sides for real and appends
  measured cost, turns, and output agreement across trials.
- **Per-node inference** — emitted judges read `ROTE_MODEL_<ID>` and
  `ROTE_BASE_URL_<ID>` at runtime, so you can swap the model or point at
  any OpenAI-compatible endpoint (Ollama, vLLM, a gateway) without
  re-emitting.

---

## The five node kinds

Every step in a graduated pipeline is exactly one of five kinds. Full
guidance:
[`references/node-kinds.md`](skills/rote-graduate/references/node-kinds.md).

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

Pick with `--runtime`; the same IR drives all of them. None of the
emitted code references MCP — the crystallization step replaces tool
calls with direct vendor API calls.

| Runtime | `--runtime` | Language | Shape | Notes |
| --- | --- | --- | --- | --- |
| **DBOS** (default) | `dbos` | Python | `main.py` — `@DBOS.workflow` + `@DBOS.step` per node | No orchestrator to deploy; SQLite (dev) / Postgres (prod) |
| Temporal | `temporal` | Python | `workflow.py` + `activities.py` | Signal handlers for HITL gates |
| Plain Python | `python` | Python | single `main.py` script | Max legibility, stdlib only; refuses HITL-gate pipelines |
| Cloudflare Workflows | `cloudflare` | TypeScript | `WorkflowEntrypoint` + `wrangler.jsonc` | `wrangler deploy`-ready |
| DBOS (TypeScript) | `dbos-ts` | TypeScript | `src/main.ts` (DBOS Transact) | Zero-orchestrator; Postgres-only |
| Inngest | `inngest` | TypeScript | one `inngest.createFunction` | Mounts into an existing Node/Next.js app; retries are function-level |

---

## Drivers

`rote` ships three interchangeable graduator drivers — pick whichever
matches your auth. The same `pipeline.yaml` comes out either way.

| Driver | Backend | Auth | Install |
| --- | --- | --- | --- |
| `claude` (default) | `claude -p` subprocess | Claude Max/Pro OAuth or `CLAUDE_CODE_OAUTH_TOKEN` | Install Claude Code separately |
| `codex` | `codex exec` subprocess | ChatGPT Plus/Pro OAuth | Install Codex CLI separately |
| `api` | `anthropic` Python SDK | `ANTHROPIC_API_KEY` | `pip install 'rote-cli[api]'` |

The `claude` driver scrubs `ANTHROPIC_API_KEY` from the subprocess so a
subscription login wins, and limits the agent to read/write/glob/grep
tools. The default model is **Sonnet** rather than Opus — the task is
structured-rubric-following, not deep reasoning, and Sonnet brings
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

`rote` is **pre-1.0**. The end-to-end flow works on the BDR example, and
all six adapters are validated by a `slow`-marked e2e suite that runs the
emitted code against real runtimes (DBOS over SQLite, Temporal's
time-skipping server, the TypeScript targets compiled with `tsc --noEmit`
and driven through both HITL gates on live dev servers, the plain-Python
script as a subprocess, and the MCP server over real stdio). The fast
suite (`pytest tests/`) makes no real API calls; the integration suite is
`pytest tests/ -m slow`.

Known gaps: the `codex` driver is a stub (`is_available` works, `run`
isn't implemented), the extracted modules are `NotImplementedError` stubs
you fill in with real API-client code, a Restate adapter is planned, and
`fan_out` nodes currently receive the whole upstream list in one
invocation (per-element dispatch is a planned enhancement). Published on
PyPI as [`rote-cli`](https://pypi.org/project/rote-cli/) via tag-driven
Trusted Publishing ([docs/releasing.md](docs/releasing.md)).

---

## Repository layout

```
rote/
├── docs/                  agent-runtime · mcp-trigger · releasing
├── skills/rote-graduate/  the graduator agent (SKILL.md + 4 reference files)
├── src/rote/
│   ├── cli.py             rote graduate / emit / eval / serve
│   ├── ir.py              Pydantic IR models + load_pipeline
│   ├── graduator/         orchestrator + drivers/ (claude · codex · anthropic_api)
│   └── adapters/          dbos · temporal · python · cloudflare · dbos_ts · inngest
│                          (+ _common / _py_common / _ts_common emit helpers)
├── examples/bdr-outreach/ source skill · hand-drafted IR baseline · run snapshots
└── tests/                 fast + slow suites (pytest -m slow)
```

---

## Documentation

- [`docs/agent-runtime.md`](docs/agent-runtime.md) — design record for the
  driver abstraction (the `claude -p` env gotcha; the non-use of
  `claude-agent-sdk`)
- [`docs/mcp-trigger.md`](docs/mcp-trigger.md) — `rote register` +
  `rote serve`: graduated pipelines as MCP tools (FastMCP 3.x)
- [`docs/releasing.md`](docs/releasing.md) — tag-driven PyPI Trusted
  Publishing
- [`skills/rote-graduate/`](skills/rote-graduate/) — the graduator's
  `SKILL.md` and its four rubric files (node kinds, crystallization
  heuristics, IR schema, LLM-judge extraction)
- [`examples/bdr-outreach/`](examples/bdr-outreach/) — the canonical
  skill, its ground-truth IR, and snapshotted real graduator runs

---

## Roadmap

In rough priority order:

1. **`CodexDriver` implementation** — same shape as `ClaudeDriver` but
   spawning `codex exec`; unlocks ChatGPT subscribers.
2. **Re-graduate BDR end-to-end with `signature_spec`** — the bundled IR
   was hand-extended with structured schemas; the rubric now teaches the
   field, but no real run has produced one yet.
3. **Pre-filter as a `pure_function` node** — today hard thresholds are
   lifted into a judge's `forward()`, which works for Temporal but not
   Cloudflare; a separate node makes the short-circuit uniform.
4. **More example skills** — BDR is one shape; research-heavy,
   retrieval-heavy, and code-review skills stress the IR differently.
5. **`fan_out` per-element dispatch** — currently the whole upstream list
   arrives in one invocation.
6. **The graduator graduating itself** — `rote-graduate` is a SKILL.md;
   pointing `rote graduate` at it should crystallize its rubric-grade
   pieces and leave only the genuinely fuzzy judgments in the loop.

---

## Contributing

The most useful contribution right now is to **run `rote graduate` on a
real skill of your own and report what happens** — the rubric was
designed against one skill and needs more. Adding a runtime adapter or a
graduator driver, or improving the rubric, are all good next steps. See
[CONTRIBUTING.md](CONTRIBUTING.md) for dev setup, the test layout, and
the adapter/driver how-tos.

## License

Apache-2.0. See [LICENSE](LICENSE).
