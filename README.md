# rote

**Graduate fuzzy AI skills into deterministic, reliable workflows.**

`rote` is a CLI that takes an Anthropic-style Skill (a `SKILL.md` plus
`references/`) and turns it into a runnable background pipeline in one
shot. An LLM agent (itself defined as a skill) reads the source skill,
applies a structured graduation rubric, and emits:

- a runtime-agnostic intermediate representation (`pipeline.yaml`),
- extracted Python modules for the deterministic parts of the skill,
- typed signature stubs for the LLM-judge parts,
- and runtime code for your durable execution engine of choice.

```sh
# Two runtimes shipped today — emit Python for Temporal, or TypeScript
# for Cloudflare Workflows:
rote graduate ./examples/bdr-outreach/skill --runtime temporal   --out ./graduated/
rote graduate ./examples/bdr-outreach/skill --runtime cloudflare --out ./graduated/
```

The name comes from *rote learning* — doing something so many times, so
reliably, that it becomes mechanical. That's what graduation does to a
skill: a fuzzy 10-20 minute agent loop becomes a deterministic pipeline
that runs in the background, costs a fraction of the tokens, and can be
regression-tested.

---

## What just happened on the bundled example

The repository includes a real BDR outreach skill (lead generation,
contact vetting, CRM upload, mandatory exclusion checks, email
personalization, manual enrollment handoff). Running `rote graduate` on
it with the default `claude` driver and Sonnet 4.6 produces:

| Output | Value |
| --- | --- |
| Total nodes in the produced IR | **22** |
| Codifiable percentage | **78.9 %** (15 of 19 non-gate nodes) |
| Extracted Python modules | **5** (`zoominfo`, `hubspot`, `conference`, `exclusions`, `report`) |
| Typed LLM-judge signatures | **2** (`vet_contact`, `personalize_email`) |
| Mandatory nodes (cannot be skipped) | **4** |
| Human-in-the-loop gates | **3** |
| Wall-clock time | ~13 minutes |
| Subscription cost | ~$0.70 (Sonnet 4.6 via Claude Code) |

The graduator independently:

- Identified the three MANDATORY exclusion checks from prose-only
  enforcement and marked them `mandatory: true` in the IR.
- Extracted four batch-size constants (`10`, `100`, `250`, `30`-day
  window) that lived only in prompt prose.
- Lifted a literal Python keyword classifier out of a reference file
  into a real module.
- Modeled a parallel "conference list" entry path with its own HITL
  gate that the human-written baseline missed.
- Surfaced five Open Questions with explicit "review ask" notes for
  the human reviewer (e.g. *"how does the adapter dispatch
  external_call nodes — via the impl Python function or via an MCP
  tool registry?"*).

After the agent finishes, a runtime adapter consumes the IR and emits
the target runtime's native code shape:

- The **Temporal** adapter emits `workflow.py` (the orchestration
  class with `@workflow.defn` and signal handlers for the HITL gates)
  and `activities.py` (one `@activity.defn` per node, lazy-importing
  the extracted functions).
- The **Cloudflare Workflows** adapter emits a TypeScript
  `WorkflowEntrypoint` class with `step.do(...)` for each unit of
  work and `step.waitForEvent(...)` for each HITL gate, plus
  `signatures/*.ts` (Zod schemas + Anthropic SDK calls) and the
  supporting `wrangler.jsonc` / `package.json` / `tsconfig.json`. The
  output is `wrangler deploy`-ready.

None of the emitted code references an MCP runtime, in either
language — the agent's crystallization step replaces tool calls with
deterministic implementations.

---

## Why this exists

Fuzzy AI skills work, but in production they're slow, expensive, and
non-deterministic:

- **Slow.** A 10-20 minute agent loop per run is fine for human-in-the-loop
  use, but unacceptable as a background job.
- **Expensive.** Multi-agent loops use ~15× the tokens of a single chat,
  per Anthropic's own measurements. Most of those tokens go to re-deriving
  procedures the skill author already wrote down.
- **Non-deterministic.** A "MANDATORY" check enforced only by prose can be
  silently skipped if prompt drift is bad or the trajectory gets long.
  There's no way to regression-test a behavior the LLM has to remember.

The fix is to identify which parts of a skill are *actually* fuzzy and
which are deterministic procedures wearing fuzzy clothing. Then move
the deterministic parts into code, keep the LLM at the points where the
input is genuinely unbounded (parsing, classifying, drafting), and wrap
the whole thing in a durable execution engine with explicit HITL gates.

That graduation step is what `rote` automates.

---

## How it works

`rote` is a three-layer system. Each layer has one job and contracts on
a small interface:

```
   ┌────────────────────┐
   │  SKILL.md +        │   Source skill bundle (untouched)
   │  references/       │
   └─────────┬──────────┘
             │
             │  rote graduate
             ▼
   ┌────────────────────┐
   │  graduator agent   │   LLM agent runs the rote-graduate
   │  (Claude / Codex / │   skill against the source bundle.
   │   Anthropic SDK)   │   Pluggable driver layer.
   └─────────┬──────────┘
             │
             │  filesystem contract:
             │    work_dir/pipeline.yaml + extracted/ + signatures/
             ▼
   ┌────────────────────┐
   │  Pipeline IR       │   Pydantic-validated DAG of typed
   │  (pipeline.yaml)   │   nodes. Five node kinds. Runtime-
   │                    │   agnostic.
   └─────────┬──────────┘
             │
             │  rote.adapters.<runtime>
             ▼
   ┌────────────────────┐
   │  emitted runtime   │   Workflow + activities for the
   │  code              │   target durable execution engine.
   └────────────────────┘
```

The three layers are:

1. **The graduator agent** (`skills/rote-graduate/`) — a regular
   Anthropic Skill with a `SKILL.md` and four reference files
   (`node-kinds.md`, `crystallization-heuristics.md`, `ir-schema.md`,
   `llm-judge-extraction.md`). This is the *brain* of `rote`. It can
   run inside any Skills-compatible surface; you don't need rote to
   use it.
2. **The IR** (`src/rote/ir.py`) — Pydantic models for the five node
   kinds plus edges, retries, HITL gates, and pipeline metadata. The
   IR is the source of truth; everything downstream is template
   substitution from it.
3. **Runtime adapters** (`src/rote/adapters/<runtime>.py`) — pluggable
   modules that consume an IR and emit runnable code for a specific
   durable execution engine.

The graduator's job ends when it has produced a valid `pipeline.yaml`
(plus extracted modules and signatures). It does not emit runtime code.
Code emission is *deterministic Python* in `rote.adapters` — never
agent-driven — so the same IR always produces byte-identical output.

---

## Quickstart

### Install

`rote` is not on PyPI yet. Clone and install in editable mode:

```sh
git clone https://github.com/trevhud/rote.git
cd rote
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

This installs `rote` plus everything you need to run the tests
(`temporalio`, `anthropic`, `pytest`, `pytest-asyncio`).

### Run on the bundled example

The repository includes a real BDR outreach skill in
`examples/bdr-outreach/skill/`. Graduate it:

```sh
rote graduate examples/bdr-outreach/skill \
  --runtime temporal \
  --out /tmp/bdr-graduated
```

By default `rote` auto-detects an available agent driver in this
order: `claude` (Claude Code CLI) → `codex` (Codex CLI) → `api`
(Anthropic SDK with `ANTHROPIC_API_KEY`). Override with `--agent`:

```sh
rote graduate examples/bdr-outreach/skill --agent api --out /tmp/bdr-graduated
```

The output directory is structured as:

```
/tmp/bdr-graduated/
├── graduated/                   # produced by the graduator agent
│   ├── pipeline.yaml            # the IR
│   ├── extracted/*.py           # deterministic functions
│   ├── signatures/*.py          # typed LLM-judge signatures
│   ├── evals/*.jsonl            # seed eval examples
│   └── graduation-report.md     # human-readable summary
└── runtime/temporal/            # produced by the adapter
    ├── workflow.py
    ├── activities.py
    └── __init__.py
```

### Render an IR without re-running the agent

If you already have a `pipeline.yaml` (hand-written or from a previous
graduation), `rote emit` runs just the adapter step:

```sh
rote emit /path/to/pipeline.yaml --runtime temporal --out /tmp/emitted/
```

This is the cheap inner loop while iterating on adapters or IR shapes.

---

## The five node kinds

Every step in a graduated pipeline is exactly one of five kinds. The
target runtime's adapter knows how to emit each one. Full classification
guidance lives in
[`skills/rote-graduate/references/node-kinds.md`](skills/rote-graduate/references/node-kinds.md).

| Kind | What it is | Where the LLM lives |
| --- | --- | --- |
| `pure_function` | Fixed logic, deterministic I/O | Not involved |
| `external_call` | Vendor API call with fixed semantics + retries | Not involved |
| `llm_judge` | Fuzzy classification against a rubric, typed I/O | Typed signature: DSPy/BAML in Python; Zod + vendor SDK in TypeScript. The IR carries a runtime-agnostic `signature_spec` (JSON Schema + prompt) so each adapter derives the right native shape. |
| `agent_loop` | Genuinely exploratory tool use | Bounded agent loop |
| `hitl_gate` | Explicit human approval, suspend until signal | Durable suspend/resume |

The guiding rule: **keep the LLM at points where the input is unbounded
or ambiguous, and codify everything else.** When a step could be
classified two ways, prefer the more deterministic kind.

---

## Driver matrix

`rote` ships three interchangeable graduator drivers. Pick whichever
matches your auth situation; the same `pipeline.yaml` comes out either
way.

| Driver | Backend | Auth | Install | Default model |
| --- | --- | --- | --- | --- |
| `claude` | `claude -p` subprocess | Claude Max/Pro OAuth or `CLAUDE_CODE_OAUTH_TOKEN` | Install Claude Code separately | `claude-sonnet-4-6` |
| `codex` | `codex exec` subprocess | ChatGPT Plus/Pro OAuth | Install Codex CLI separately | (driver default) |
| `api` | `anthropic` Python SDK | `ANTHROPIC_API_KEY` env var | `pip install rote[api]` | `claude-sonnet-4-6` |

The `claude` driver is the default for subscription users — it scrubs
`ANTHROPIC_API_KEY` from the subprocess environment so the user's
Claude Code login wins, sets
`CLAUDE_CODE_DISABLE_NONINTERACTIVE_ANIMATIONS=1` for clean output,
and limits the agent to read/write/glob/grep tools (no shell, no
network). See [`docs/agent-runtime.md`](docs/agent-runtime.md) for the
full design record including the auth gotcha that motivates the env
scrub.

The model defaults to **Sonnet 4.6** rather than Opus because the
graduator's task is structured-rubric-following, not deep reasoning.
Sonnet brings per-run cost from ~$3.50 to ~$0.70 in subscription
accounting, which makes iterative rubric tuning feasible. Override
with `rote graduate --model claude-opus-4-6` for complex skills where
Opus's extra reasoning earns its cost.

---

## Status

`rote` is **pre-1.0**. The end-to-end flow works on the BDR example
and the test suite covers each layer (133 tests in the fast suite,
plus 3 slow tests that compile the emitted Cloudflare TypeScript
against the real `@cloudflare/workers-types` definitions and drive a
real workflow instance through both HITL gates via `wrangler dev`).
Run `pytest tests/` for the fast suite; `pytest tests/ -m slow` for
the toolchain-dependent integration tests.

| Component | Status |
| --- | --- |
| IR (Pydantic schema, validation, YAML loader) | working |
| Temporal adapter | working (validated with mocked-activities e2e test) |
| Cloudflare Workflows adapter | working (validated with `tsc --noEmit` over the real emitted output) |
| Graduator orchestrator | working |
| `rote graduate` / `rote emit` CLI commands | working |
| `claude` driver | working |
| `api` (Anthropic SDK) driver | working |
| `codex` driver | stub (`is_available` works; `run` not implemented) |
| Inngest / Restate / DBOS adapters | planned |
| Real implementations of the extracted modules | the agent produces stubs that raise `NotImplementedError`; humans fill them in with real API client code |
| Workflow data flow between activities | empty payloads in v0; explicit input/output threading is a planned enhancement |
| Distribution via PyPI | not yet — install from source |

The project explicitly **does not** depend on `claude-agent-sdk`.
Anthropic's terms of service forbid third-party agents built on the
Agent SDK from using claude.ai login credentials without prior
approval, which would defeat the subscription path. We use the bare
`anthropic` SDK or spawn `claude` directly instead.

---

## Repository layout

```
rote/
├── README.md
├── LICENSE                                  # Apache-2.0
├── pyproject.toml                           # rote + optional [temporal] / [api] / [dev] extras
├── docs/
│   └── agent-runtime.md                     # decision record for the driver layer
├── skills/
│   └── rote-graduate/
│       ├── SKILL.md                         # the graduator agent's instructions
│       └── references/
│           ├── node-kinds.md                # 5-kind classification rubric
│           ├── crystallization-heuristics.md  # patterns for moving prose into code
│           ├── ir-schema.md                 # pipeline.yaml reference
│           └── llm-judge-extraction.md      # how to design typed signatures
├── src/rote/
│   ├── cli.py                               # rote graduate / rote emit
│   ├── ir.py                                # Pydantic IR models + load_pipeline
│   ├── graduator/
│   │   ├── __init__.py                      # Graduator orchestrator
│   │   └── drivers/
│   │       ├── __init__.py                  # Protocol + registry + auto_detect
│   │       ├── claude.py                    # ClaudeDriver (subprocess)
│   │       ├── codex.py                     # CodexDriver (stub)
│   │       └── anthropic_api.py             # AnthropicApiDriver (in-process)
│   └── adapters/
│       ├── __init__.py                      # adapter registry
│       ├── temporal.py                      # TemporalAdapter (Python emitter)
│       └── cloudflare.py                    # CloudflareAdapter (TypeScript emitter)
├── examples/
│   └── bdr-outreach/
│       ├── skill/                           # the source skill (graduator input)
│       ├── expected/                        # hand-drafted IR + stubs (regression baseline)
│       └── runs/                            # snapshots of real graduator runs
└── tests/                                   # 136 passing tests across 11 files
```

---

## How it differs from other tools

- **vs. raw Temporal / Cloudflare Workflows / Inngest / Restate:**
  durable execution engines give you the workflow runtime; they don't
  help you decide *what should be a workflow in the first place*.
  `rote` is the missing step that converts a working skill into
  something worth running on a durable engine.
- **vs. LangGraph:** LangGraph is an excellent state machine for
  agent loops, but its graph is hand-built. `rote` produces a graph
  from prose, classifies its nodes by determinism, and pushes work
  out of the agent loop wherever the data supports it.
- **vs. just using Claude Code Skills directly:** Skills run great in
  interactive use. `rote` is what you reach for when a skill becomes
  business-critical and needs to run unattended in the background
  with hard reliability guarantees and per-step regression tests.
- **vs. `claude-agent-sdk`:** see the *Status* section. The Agent SDK
  is API-key-only for third-party tooling per Anthropic's ToS, which
  defeats the subscription path that `rote`'s primary `claude`
  driver enables.

---

## Documentation index

- [`docs/agent-runtime.md`](docs/agent-runtime.md) — design record for
  the driver abstraction, including the `claude -p` env-var gotcha and
  the explicit non-use of `claude-agent-sdk`
- [`skills/rote-graduate/SKILL.md`](skills/rote-graduate/SKILL.md) —
  the graduator agent's procedural instructions (the "brain")
- [`skills/rote-graduate/references/node-kinds.md`](skills/rote-graduate/references/node-kinds.md) —
  the 5-kind classification rubric with BDR examples
- [`skills/rote-graduate/references/crystallization-heuristics.md`](skills/rote-graduate/references/crystallization-heuristics.md) —
  the seven patterns for moving prose into deterministic code
- [`skills/rote-graduate/references/ir-schema.md`](skills/rote-graduate/references/ir-schema.md) —
  the `pipeline.yaml` reference (matches `src/rote/ir.py`)
- [`skills/rote-graduate/references/llm-judge-extraction.md`](skills/rote-graduate/references/llm-judge-extraction.md) —
  how to turn a fuzzy rubric into a typed signature
- [`examples/bdr-outreach/`](examples/bdr-outreach/) — the canonical
  source skill, the hand-drafted ground-truth IR, and snapshotted real
  graduator runs

---

## Roadmap

In rough priority order:

1. **`CodexDriver` implementation.** Same shape as `ClaudeDriver` but
   spawning `codex exec`. Unlocks ChatGPT subscribers.
2. **End-to-end re-graduation of BDR with `signature_spec`.** The
   current bundled `pipeline.yaml` was hand-extended with structured
   schemas for the Cloudflare adapter; the rubric in
   `skills/rote-graduate/references/` was updated to teach the
   graduator the new field, but no real graduator run has produced
   one yet. Re-running `rote graduate examples/bdr-outreach/skill`
   should produce the structured form natively.
3. **A third runtime adapter.** Probably Inngest, since its
   programming model is meaningfully different from both Temporal and
   Cloudflare. Each new adapter is also a stress test on whether the
   IR is genuinely runtime-agnostic vs. accidentally shaped like one
   of the existing targets.
4. **Pre-filter as `pure_function` node.** Today the rubric lifts
   hard thresholds into a Python `forward()` method, which works for
   Temporal but not for Cloudflare. Modeling the pre-filter as a
   separate `pure_function` node before the `llm_judge` makes the
   short-circuit work uniformly across runtimes.
5. **Explicit data-flow threading.** Both adapters currently pass
   empty payloads between steps. Real production usage needs typed
   payloads derived from each node's `input:` and `output:` schema.
6. **More example skills.** BDR is rich but it's one shape of skill.
   Additional examples (research-heavy, retrieval-heavy, code-review)
   stress-test the IR and the rubric in different ways.
7. **PyPI distribution.** Once the API is stable enough.
8. **The graduator graduating itself.** The `rote-graduate` skill is
   itself a SKILL.md. Pointing `rote graduate` at it should produce a
   graduated meta-graduator where the rubric-grade pieces are
   crystallized into Python and only the genuinely fuzzy judgments
   stay in the agent loop.

---

## Contributing

The most useful contributions right now are:

- **Run `rote graduate` on a real skill of your own and report what
  happens.** The rubric was designed against one skill (BDR); it
  needs to be tested against more.
- **Add a runtime adapter.** The Temporal adapter in
  `src/rote/adapters/temporal.py` is ~450 lines and follows a clear
  pattern. Inngest, Restate, DBOS, and Hatchet are all good targets.
- **Add a graduator driver.** The Protocol in
  `src/rote/graduator/drivers/__init__.py` is simple. Aider, Gemini
  CLI, and Cursor Agent are reasonable additions.
- **Improve the rubric.** Every change to a file under
  `skills/rote-graduate/references/` is tracked in git, so improvements
  can be A/B tested across runs.

The test suite (`pytest tests/`) covers each layer in isolation plus
the full pipeline against the BDR example. New work should land with
matching tests.

---

## License

Apache-2.0. See [LICENSE](LICENSE).
