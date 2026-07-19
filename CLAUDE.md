# CLAUDE.md — context for AI collaborators

This file is for Claude Code (and similar coding agents) working in
the `rote` repository. Read it before making non-trivial changes.
It captures the project's mental model, architectural invariants,
gotchas, and the canonical examples you should imitate.

If you're a human reading this, you probably want
[README.md](README.md) or [CONTRIBUTING.md](CONTRIBUTING.md)
instead — this file is deliberately oriented toward an AI that
needs a fast mental model before it starts editing code.

---

## What rote is

`rote` is a CLI that takes an Anthropic-style Skill (a `SKILL.md` +
`references/`) and graduates it into a runnable background pipeline:

1. An LLM agent (itself defined as a skill at
   [`skills/rote-graduate/`](skills/rote-graduate/)) reads the
   source skill and applies a structured rubric.
2. The agent produces a `pipeline.yaml` (runtime-agnostic IR) plus
   extracted Python modules and typed LLM-judge signatures.
3. A runtime adapter (DBOS by default; Temporal, Cloudflare
   Workflows, DBOS-TS, and Inngest too; more planned) consumes the
   IR and emits the runtime's native code shape — a single `main.py`
   DBOS app, a `workflow.py` + `activities.py` pair for Temporal, a
   TypeScript `WorkflowEntrypoint` class plus `wrangler.jsonc` /
   `package.json` for Cloudflare, or an Inngest function app that
   mounts into an existing Node/Next.js service.

The result is a deterministic workflow that replaces a fuzzy
10-20 minute agent loop. That's the whole product in one paragraph.

---

## Three-layer architecture

Every piece of code in this repo belongs to exactly one layer.
**Respect the layer boundaries** — most of the project's value
comes from keeping them clean.

### Layer 1 — The graduator agent (the "brain")

- **Lives at:** [`skills/rote-graduate/`](skills/rote-graduate/)
- **Consists of:** a `SKILL.md` plus five rubric files
  (`node-kinds.md`, `crystallization-heuristics.md`, `ir-schema.md`,
  `llm-judge-extraction.md`, `implementation.md`)
- **What it does:** classifies every step of a source skill into
  one of 5 node kinds, extracts deterministic procedures into
  Python modules, designs typed signatures for LLM-judge steps, and
  assembles the whole thing into a `pipeline.yaml`
- **Runs under:** a pluggable driver layer (see Layer 2)

**When you edit this layer:** you are changing how the agent
behaves. Changes are testable by running `rote graduate` end-to-end
against `examples/bdr-outreach/skill` and diffing the output
against the previously committed snapshot in
`examples/bdr-outreach/runs/`.

### Layer 2 — The driver layer (the "runtime for the agent")

- **Lives at:** [`src/rote/graduator/drivers/`](src/rote/graduator/drivers/)
- **Protocol:** every driver implements `GraduatorDriver` with
  `name`, `is_available() -> (bool, str)`, and
  `async run(skill_dir, graduator_skill_dir, work_dir) -> DriverResult`
- **Three concrete drivers:** `ClaudeDriver` (subprocess to `claude -p`),
  `CodexDriver` (subprocess to `codex exec`), `AnthropicApiDriver`
  (in-process via the `anthropic` SDK)
- **The contract is the filesystem:** every driver writes
  `work_dir/pipeline.yaml` and returns a `DriverResult` pointing at it.
  Do not invent new return-shape fields or side-channel comms.

**When you edit this layer:** you are changing how the agent is
invoked, not what it does. Changes are testable by mocking
subprocess spawning (see
[`tests/test_claude_driver.py`](tests/test_claude_driver.py)) or
the Anthropic SDK client (see
[`tests/test_anthropic_driver.py`](tests/test_anthropic_driver.py)).

### Layer 3 — The runtime adapters (the "code emitter")

- **Lives at:** [`src/rote/adapters/`](src/rote/adapters/)
- **Five adapters today:**
  - `DbosAdapter`
    ([`src/rote/adapters/dbos.py`](src/rote/adapters/dbos.py))
    — the CLI default; emits a single durable Python app (`main.py`
    with `@DBOS.workflow`/`@DBOS.step`, plus `signatures/*.py`,
    `extracted/*.py`, `dbos-config.yaml`, `README.md`) — no
    orchestrator process, state in SQLite/Postgres
  - `TemporalAdapter`
    ([`src/rote/adapters/temporal.py`](src/rote/adapters/temporal.py))
    — emits Python (`workflow.py` + `activities.py`)
  - `CloudflareAdapter`
    ([`src/rote/adapters/cloudflare.py`](src/rote/adapters/cloudflare.py))
    — emits TypeScript (`src/workflow.ts` extending
    `WorkflowEntrypoint`, plus `signatures/*.ts`, `extracted/*.ts`,
    and a `wrangler.jsonc` / `package.json` / `tsconfig.json`)
  - `PythonAdapter`
    ([`src/rote/adapters/python.py`](src/rote/adapters/python.py),
    registered as `python`) — the maximum-legibility target: emits a
    plain, orchestrator-free Python script (`main.py` with one function
    per node, visible inline retry loops, stdlib `ThreadPoolExecutor`
    for parallel waves, plus `extracted/`, `signatures/`,
    `requirements.txt`). **Refuses pipelines with `hitl_gate` nodes at
    emit time** (via the derived `Pipeline.requires_durable_execution`
    property) — a plain script cannot durably park for human approval;
    the error points at `--runtime dbos`. Shares the Pydantic
    signature / extracted-stub emitters with the DBOS adapter via
    [`src/rote/adapters/_py_common.py`](src/rote/adapters/_py_common.py).
  - `DbosTsAdapter`
    ([`src/rote/adapters/dbos_ts.py`](src/rote/adapters/dbos_ts.py),
    registered as `dbos-ts`) — emits TypeScript for DBOS Transact
    (`src/main.ts` with `DBOS.registerWorkflow` / `DBOS.registerStep`
    function wrappers, plus `signatures/*.ts`, `extracted/*.ts`, and
    `package.json` / `tsconfig.json` / `dbos-config.yaml`). Shares the
    TS emission machinery in
    [`src/rote/adapters/_ts_common.py`](src/rote/adapters/_ts_common.py)
    with the Cloudflare adapter. Note: the DBOS **TS** SDK is
    Postgres-only (no SQLite parity with DBOS Python), and its
    `DBOS.recv` defaults to a 60s timeout — the emitter always passes
    the IR timeout explicitly.
  - `InngestAdapter`
    ([`src/rote/adapters/inngest.py`](src/rote/adapters/inngest.py),
    registered as `inngest`) — emits a TypeScript Inngest app (one
    durable `inngest.createFunction` in `src/inngest/pipeline.ts`
    running the DAG waves, `step.waitForEvent` per HITL gate on a
    `<pipeline>/<signal>` event, plus a framework-neutral
    `inngest/node` serve entrypoint and a README documenting the
    Next.js mount). Shares `_ts_common.py` with the other TS
    adapters. Note: Inngest v4 retries are **function-level only** —
    the emitter maps max-across-nodes and comments the per-node
    deltas (see the Inngest gotcha below).
- **What they do:** consume a validated `Pipeline` IR and write
  runtime-native code into an output directory
- **Never run an agent loop.** Code emission is pure template
  substitution. The agent's job ends when the IR is valid; the
  adapter's job begins there.

**When you edit this layer:** you are changing the output code
format or adding a new runtime. Changes are testable with unit
tests covering emission (AST / textual invariants), plus a real
runtime smoke test — `tests/test_temporal_e2e.py` for Temporal
(time-skipping `WorkflowEnvironment`); `tests/test_cloudflare_e2e.py`
for Cloudflare (runs `npm install` + `tsc --noEmit` on the emitted
output, gated by `@pytest.mark.slow`); `tests/test_dbos_ts_e2e.py`
for DBOS TypeScript (`npm install` + `tsc --noEmit`, plus a live run
on the real DBOS TS runtime against a Docker Postgres);
`tests/test_inngest_e2e.py` for Inngest (`npm install` +
`tsc --noEmit`, plus a live run through both HITL gates against the
real Inngest dev server, `inngest-cli dev`).

---

## The IR is the source of truth

[`src/rote/ir.py`](src/rote/ir.py) defines the Pydantic models that
every layer agrees on. If you're uncertain about a field's semantics,
read that file — it's ~290 lines, fully typed, and the authoritative
reference (not the rubric's `ir-schema.md`, which is agent-facing
prose that can drift).

**The five node kinds:**

| Kind | Required field | Where the LLM lives |
| --- | --- | --- |
| `pure_function` | `impl` | Not involved |
| `external_call` | `impl` | Not involved |
| `llm_judge` | `signature` (legacy path) and/or `signature_spec` (structured) | Typed signature (DSPy / BAML in Python; Zod + vendor SDK in TS) |
| `agent_loop` | `tools` | Bounded agent loop |
| `hitl_gate` | `signal` | Durable suspend/resume |

**The `signature_spec` model** ([`src/rote/ir.py`](src/rote/ir.py))
carries a JSON Schema for input + output, a Jinja-style prompt
template, and the LLM client config (anthropic / openai). It's the
runtime-agnostic alternative to the legacy `signature: 'path/to/file.py:Class'`
form, which only works for Python adapters. Cloudflare requires
`signature_spec`; Temporal accepts either and prefers `signature_spec`
when present.

**The canonical skill that covers all 5 kinds:** the BDR outreach
example in `examples/bdr-outreach/skill/`. If you need an IR that
exercises every code path, its hand-drafted baseline is at
`examples/bdr-outreach/expected/pipeline.yaml`.

---

## Gotchas (things you will absolutely trip on if you don't read this)

### `claude -p` and the `ANTHROPIC_API_KEY` trap

Claude Code's print mode (`claude -p`) has a documented behavior
where `ANTHROPIC_API_KEY` **always wins** over any active OAuth
session. If your system has the env var set, `claude -p` will use
API-key billing even though the user expected subscription auth.

**The fix:** `ClaudeDriver._build_child_env()` scrubs
`ANTHROPIC_API_KEY` and `ANTHROPIC_AUTH_TOKEN` from the child
environment before spawning. **Do not remove this.** If you need
API-key auth to Anthropic, use `AnthropicApiDriver` instead — it's
a separate driver precisely so the Claude driver can force
subscription auth.

### YAML 1.1 parses `on:` as boolean `True`

PyYAML follows YAML 1.1, which has historical bool keys (`on`,
`off`, `yes`, `no`). A field literally named `on` in a YAML file
becomes the Python value `True`, not the string `"on"`. This
**broke the first IR test run** when I tried to use `on:` for
retry trigger conditions.

**The fix:** the IR uses `retry_on:` (with the `_on` suffix), not
`on:`. Propagate this convention if you add new fields.

### Loop body sub-nodes are excluded from top-level waves

An `agent_loop` node can declare a `loop_body: [node_id, ...]`
listing sub-nodes invoked per iteration. Those sub-nodes also
exist as top-level entries in the `nodes:` list (so they can be
tested in isolation) but the adapter's `_execution_waves` function
filters them out of the top-level DAG — otherwise the workflow
would dispatch them twice. See `rote.adapters.temporal._execution_waves`
for the reference implementation.

### Driver factories accept `**kwargs`

The driver registry's factories have signature `**kwargs ->
GraduatorDriver` because `Graduator(model=...)` needs to pass
kwargs through at construction time. Drivers that don't care about
a particular kwarg should swallow it via `**kwargs` in their
`__init__`, not hard-fail. **Do not change the factory signature
to typed keyword args** — that breaks the registry's polymorphism.

### Emitted code touches MCP only through an explicit `mcp:` binding

Two regimes, don't mix them up:

- **Nodes without an `mcp:` binding must never reference MCP** in any
  adapter or language — no `mcp` imports, no `@mcp.tool()` decorators,
  no MCP SDK dependencies. Enforced per adapter:
  Temporal — `tests/test_temporal_adapter.py::test_emitted_activities_never_reference_mcp`
  (Python AST walk); Cloudflare —
  `tests/test_cloudflare_adapter.py::test_emitted_files_never_reference_mcp`
  (every `.ts` file, comments + string literals stripped). Copy the
  matching test when adding an adapter.
- **Nodes WITH an `mcp:` binding** (DBOS adapter, `--backend mcp`, the
  default) emit a *working* FastMCP call: the step goes through the
  emitted `extracted/_rote_mcp.py` helper — the verbatim source of
  `rote.mcp._runtime_helper` (never hand-edit the emitted copy; fix
  the module) — which resolves the endpoint (env > rote registry > IR)
  and credentials (`rote mcp login` token store > registry static
  headers > unauthenticated) at runtime. On DBOS Python AND DBOS-TS,
  auth failures **park the workflow durably** instead of failing it
  (`RoteMcpAuthNeeded` → `DBOS.recv` on `rote:auth:<server>`;
  `rote mcp login` — or `rote mcp release` without a login — discovers
  parked workflows via the app registry + the `rote_auth_status` event
  and releases them, one Python code path for both runtimes). Traps if
  you touch this: a parked DBOS workflow is just PENDING (no WAITING
  status — discovery *requires* the advertised event); parallel-wave
  siblings can surface stale auth failures after the release signal was
  already consumed — hence the retry-once-before-parking shape in the
  emitted wrappers; and the cross-language leg only works in DBOS's
  *portable* serialization — the emitted TS `setEvent` passes
  `{ serializationType: "portable" }` and the Python release `send`
  passes `WorkflowSerializationFormat.PORTABLE` (the defaults,
  superjson and pickle, are mutually unreadable). In TS, detect the
  auth error by `name` string, never `instanceof` (serialized errors
  reconstruct as plain `Error`s). See
  [`docs/mcp-client.md`](docs/mcp-client.md).

### Sonnet is the default, not Opus

Both `ClaudeDriver` and `AnthropicApiDriver` default to
`claude-sonnet-4-6` rather than Opus, because Opus is ~5× more
expensive and Sonnet is fully capable of following the structured
rubric. We learned this the hard way — the first two BDR runs with
Opus exhausted a Claude Max/Pro "extra usage" budget in two
attempts (~$3.50 each). Don't "fix" the default to Opus unless you
have specific evidence that a skill needs it.

### `DEFAULT_MAX_TURNS = 60`

BDR-scale skills need ~25 tool calls minimum, realistically 40-50
with exploration. The initial default of 30 caused an
`error_max_turns` failure on the first real run. 60 leaves headroom.
Don't reduce it without measuring.

### Cloudflare local-dev `status` stays `"running"` while parked on `waitForEvent`

The Cloudflare e2e test (`tests/test_cloudflare_e2e.py`) needs to know
when a workflow instance has parked on a `step.waitForEvent` so it
can deliver the resume signal. The intuitive check —
`status.status === "waiting"` — does **not** work in local dev. The
top-level `status` field stays `"running"` the entire time; only the
human-readable `wrangler workflows instances describe` output
distinguishes "💤 Waiting for event".

**The fix:** the test infers parking by step-output stability. Once
`__LOCAL_DEV_STEP_OUTPUTS` reaches the expected count *and* stops
growing for a few consecutive polls, the workflow is necessarily
either at a HITL gate or done. See `_wait_until_parked_or_complete`
in the e2e test for the canonical implementation. Production behavior
may differ — this is a local-dev quirk; document it if anything
changes upstream.

Related: `__LOCAL_DEV_STEP_OUTPUTS` includes both `step.do` results
*and* `waitForEvent` payloads in the same array. Don't rely on the
length matching node count; compare node-name **sets** instead.

### Cloudflare `step.do` rejects `Record<string, unknown>` returns

`step.do`'s generic is `T extends Rpc.Serializable<T>`, and
`Record<string, unknown>` does **not** satisfy it — `unknown` values
aren't structurally serializable, so annotating an emitted stub with
that return type breaks overload resolution at every `step.do` call
site. Leaving the return inferred is no better: a throwing function
declaration infers `Promise<void>`, and `void as Record<...>` fails
TS2352 at the data-flow reference sites.

**The fix:** emitted stubs declare `Promise<never>` (honest for an
always-throwing stub; `never` satisfies any constraint and casts to
anything), and the workflow emits node-output field access as
`(<id>_result as Record<string, unknown>)["field"]` so it compiles
against both the stubs and whatever concrete type the user fills in
later. Verified by `tests/test_cloudflare_e2e.py::test_emitted_typescript_compiles`
— run it after touching the Cloudflare emitter's typing story.

### Inngest v4: no per-step retries, no 3-arg `createFunction`, and a lying dev-server endpoint

Three traps hit while building the Inngest adapter, all verified
empirically against `inngest` 4.11.0 + `inngest-cli` 1.34.0:

1. **`createFunction` is two-argument in v4.** Triggers live in the
   options object: `createFunction({ id, retries, triggers: [{ event }] },
   handler)`. The classic `createFunction(opts, { event }, handler)` form
   from docs/blog posts **throws at runtime**. tsc alone won't catch a
   regression to the old form in emitted-string templates — the e2e's
   live run does.
2. **Per-step retry/timeout config does not exist.** `StepOptions` is
   `{ id, name?, parallelMode? }`. The IR's per-node `RetryPolicy` maps
   to a single function-level `retries` (max across nodes, clamped 0–20);
   per-node deltas, `backoff`, `retry_on`, and per-node `timeout` are
   emitted as comments. Don't "fix" the adapter to pass per-step retry
   options — there's nowhere to put them.
3. **`GET /v1/events/{id}/runs` lies while a run is parked.** During
   probing it reported `status: "Completed"` while the run was parked on
   `step.waitForEvent`. Use it only to discover the `run_id`, then poll
   `GET /v1/runs/{run_id}` (truthful `Running`/`Completed`). The run's
   *return value* is only available via the dev server's GraphQL API
   (`POST /v0/gql`, `run(runID:){output}` → `RunComplete` op JSON); the
   v1 REST `output` field comes back empty. Also: with `--no-poll` the
   dev server's single startup sync can race the app boot and never
   register it — the e2e sends an explicit `PUT` to the app's serve
   handler to force registration. See `tests/test_inngest_e2e.py`.

### Eval pricing: no hardcoded prices, and two traps found empirically

`rote.eval.pricing` fetches current models + official prices at eval
time (models.dev primary, OpenRouter cross-check; 24h cache in
`~/.cache/rote/pricing.json`; loud `PricingError` when offline — never
a baked-in table). Two things you will get wrong if you touch it:

1. **"Most expensive = flagship" is false.** Previous-generation
   flagships linger in catalogs at higher prices than the current one
   (Opus 4.1 at $15/Mtok vs Fable 5 at $10/Mtok). Flagship detection
   restricts to the provider's *current generation* — models released
   within `_CURRENT_GENERATION_WINDOW` of the provider's own newest
   release date. The anchor is the data's newest release, never the
   wall clock, so tier detection is a pure function of the payload and
   testable with fixtures. Regression:
   `tests/test_eval_pricing.py::test_tier_detection_matches_current_lineup`.
2. **Community catalogs lag launches.** On Sonnet 5's launch week,
   LiteLLM's price file still carried the 4.6-generation price while
   models.dev and OpenRouter agreed on the new one — that's why
   LiteLLM was dropped and why the cross-check annotates the
   scorecard's `source` string with a WARNING on >2% disagreement
   instead of trusting either side silently.

Related invariant: the eval sidecar (`eval.yaml`, per-step agent-turn
estimates written by the graduator) is deliberately NOT part of the IR
— it describes the *source skill's* behavior as an agent, not the
pipeline. Don't move it into `pipeline.yaml`; invariant #1 (the IR is
runtime-agnostic and behavior-only) applies.

### ClaudeDriver recovers `pipeline.yaml` if subprocess errors after writing

The agent's deliverable is a file on disk, not a clean exit code. A
real-world failure mode hit during this project: the agent wrote a
complete `pipeline.yaml` at turn N, then ran a self-validation turn
at N+1 that hit `ECONNRESET` — the subprocess returned nonzero, the
orchestrator's `TemporaryDirectory` cleanup ran, ~$2.50 of Sonnet 4.6
work disappeared.

The driver now checks for `pipeline.yaml` **before** failing on
returncode. If the file exists, the run succeeded materially; the
subprocess error is surfaced in `metadata["subprocess_warning"]`
rather than treated as fatal. Don't undo this — losing $2 runs to a
network blip is unacceptable. See
`tests/test_claude_driver.py::test_nonzero_exit_with_pipeline_yaml_recovers_run`.

---

## Canonical examples to imitate

When adding new code, the fastest way to stay consistent is to
copy the shape of an existing canonical example.

| What you want to do | Imitate |
| --- | --- |
| Add a new node kind to the IR | `rote.ir.Node` — add an enum variant, a kind-specific required field, and a branch in `_validate_kind_specific_fields` |
| Add a Python-emitting runtime adapter | `src/rote/adapters/temporal.py` (~570 lines) |
| Add a TS-emitting runtime adapter | `src/rote/adapters/cloudflare.py` (~750 lines) — includes the JSON-Schema-to-Zod converter and signal-name validator |
| Add a subprocess-based driver | `src/rote/graduator/drivers/claude.py` + `tests/test_claude_driver.py` |
| Add an in-process driver | `src/rote/graduator/drivers/anthropic_api.py` + `tests/test_anthropic_driver.py` |
| Add a new CLI subcommand | `_cmd_emit` in `src/rote/cli.py` (simple) or `_cmd_graduate` (which calls into the Graduator orchestrator) |
| Test a Python-emitting adapter end-to-end | `tests/test_temporal_e2e.py` (uses Temporal's `WorkflowEnvironment.start_time_skipping`) |
| Test a TS-emitting adapter end-to-end | `tests/test_cloudflare_e2e.py` (real `npm install` + `tsc --noEmit`; `@pytest.mark.slow`) |
| Test the CLI via subprocess | `tests/test_cli.py::test_subprocess_emit_bdr` |
| Add a rubric file | Any file in `skills/rote-graduate/references/` — keep them 200-300 lines, concrete, with BDR examples for every point |

---

## Invariants that must hold

These are the things that, if violated, break the project's design.
If a change of yours would violate any of them, stop and reconsider.

1. **The IR is runtime-agnostic.** No Temporal-specific or
   Inngest-specific concepts in `rote.ir`. If a field is meaningful
   only to one runtime, it belongs in that adapter's config, not
   the IR.
2. **Drivers contract on the filesystem only.** `work_dir/pipeline.yaml`
   is the deliverable. No stdout parsing, no side channels.
3. **Emitted code touches MCP only through an explicit `mcp:`
   binding.** Nodes without a binding never reference MCP in any
   adapter or language (enforced by AST/text tests); nodes with one
   emit a working, *never-interactive* client call — auth problems
   raise/park, they never open a browser from workflow code.
4. **Mandatory nodes cannot become conditional.** The IR validator
   rejects `mandatory: true` on `agent_loop` nodes, and the
   Temporal adapter emits mandatory nodes as unconditional
   activities. Adding a conditional-skip mechanism would destroy
   the "MANDATORY prose check becomes impossible-to-skip code"
   value prop.
5. **The repo is PH-free and secret-free.** `scripts/sanity-check.sh`
   is the source of truth. Run it before any push. If you add a
   new example skill or a new test fixture, run the script
   afterward.
6. **Two adapters before declaring the IR generic.** *(Satisfied as
   of v0.2.0.)* The IR was stress-tested by the Cloudflare adapter,
   which emits TypeScript with a single-class programming model
   (vs. Temporal's workflow + activities split). Two pressure points
   surfaced:

   - The retry `backoff` enum already maps cleanly to Cloudflare's
     categorical enum — no IR change needed.
   - HITL signal names need a `[A-Za-z0-9_-]+` constraint for
     Cloudflare's `waitForEvent`, but this is enforced at adapter
     emission time, not in the IR.
   - LLM signatures needed a structured cross-language form — solved
     by adding `signature_spec` (JSON Schema + prompt) alongside the
     legacy Python-path `signature` field.

   The Inngest adapter added a third pressure point without requiring
   an IR change: Inngest v4 has **no per-step retry/timeout config**,
   so the per-node `RetryPolicy` degrades to a function-level budget
   (max across nodes) with per-node deltas emitted as comments. That's
   a lossy mapping documented at the adapter layer — acceptable
   because the IR stays the richer form; a runtime that can't express
   a policy documents the gap rather than forcing the IR down to the
   lowest common denominator.

   Future adapters (Restate, etc.) shouldn't require IR changes for
   their core programming model — but if one does, that's a real
   signal to revisit the IR shape rather than papering over it in the
   adapter.

7. **IR string fields that reach emitted code are charset-constrained
   at the IR boundary, not in the adapters.** A `pipeline.yaml` is
   untrusted input (it's LLM-generated from a possibly third-party
   `SKILL.md`, or shared directly), yet every adapter splices certain
   fields *verbatim* into emitted source and filenames. `Node.id`
   (emitted as `async def {id}` / `@activity.defn(name=…)` and as
   `signatures/{id}.py`), `signal` (a Temporal signal-handler method /
   DBOS topic), and the two halves of `impl`/`signature`
   (`from … import {symbol}` + call site, and the module path) are all
   validated to a safe identifier / relative-path shape by
   `field_validator`s in `rote.ir` — see `_IDENTIFIER_RE`,
   `_validate_impl_ref`, and the `Node._validate_id/_validate_signal/
   _validate_impl_signature` validators. Prose fields (`description`)
   can't be charset-restricted, so they're escaped at emission via
   `rote.adapters._common.safe_docstring_line` (Python docstrings) and
   `safe_block_comment_line` (TS JSDoc). Emitted-file writes also pass
   through `resolve_within` as defense-in-depth. **Do not add a new IR
   string field that an adapter emits verbatim without giving it an
   equivalent validator**, and don't move these checks down into a
   single adapter — the IR boundary is where all three (and every
   future) adapter inherit them. Regression coverage:
   `tests/test_ir.py::test_node_id_must_be_a_safe_identifier`,
   `::test_impl_symbol_injection_is_rejected`, and
   `tests/test_temporal_adapter.py::test_malicious_description_cannot_break_out_of_docstring`.

---

## Workflow expectations

### Running tests

```sh
.venv/bin/pytest tests/                  # full suite, ~1s
.venv/bin/pytest tests/test_ir.py        # fast iteration on IR changes
```

The suite is intentionally fast (no real API calls, no real
subprocess spawns, no real LLM invocations). Every slow thing is
mocked. Keep it that way — if a new test needs to be slow, gate
it behind a `@pytest.mark.slow` decorator and document why.

### Running the real graduator

```sh
.venv/bin/rote graduate examples/bdr-outreach/skill \
  --runtime temporal \
  --out /tmp/bdr-graduated
```

~13 minutes wall clock on BDR. Expect 30-40 turns with Sonnet 4.6.
The output landing at `/tmp/bdr-graduated` (or wherever you point
`--out`) is what you'd commit as a new regression snapshot if the
rubric or IR changed in a way that requires re-graduating.

### Sanity-checking before commit

```sh
./scripts/sanity-check.sh
```

Must exit 0. If it exits non-zero, either the scan found real leaks
(fix them) or the pattern needs adjustment (be careful — err on
the side of false positives over false negatives).

### Before a PR

- [ ] `pytest tests/` — all tests pass
- [ ] `./scripts/sanity-check.sh` — clean
- [ ] `ruff check . && ruff format .`
- [ ] `mypy src/rote` — strict, no ignores
- [ ] If the rubric or IR changed materially: re-run the graduator
      on BDR and diff the snapshot
- [ ] Commit message explains the *why*, not just the *what*

---

## What's stubbed vs. working

Don't waste time debugging stubs. These are intentional.

**Intentionally stubbed:**

- The BDR example's `extracted/*.py` modules raise
  `NotImplementedError` — users fill them in with real API client
  code; the graduator produces scaffolding, not production code
- `fan_out` nodes receive the whole upstream list in one invocation —
  per-element dispatch is a planned enhancement on top of data-flow
  threading

**Working end-to-end:**

- IR load + validate
- `rote emit` (IR → runtime code; DBOS default), with hash-guarded
  re-emission: every adapter writes through
  `rote.adapters._common.EmitWriter`, which records what rote wrote in
  `.rote-manifest.json` and, on re-emit, preserves any file the user
  edited (fresh content lands in a `<name>.new` sibling and the CLI
  reports it). Never write emitted files with bare `write_text` in an
  adapter — route through the writer.
- `rote graduate` (SKILL.md → IR → Temporal code), stamping a
  `provenance.json` sidecar (section hashes; see `rote.skill_source`)
- `rote graduate --update` — incremental re-graduation: diffs the
  skill against the stamped provenance, re-derives only nodes whose
  source sections changed, enforces that unchanged nodes' ids survive,
  and merges file-level so user-filled stubs are kept. No section
  changes → no agent run at all.
- Per-node inference overrides in emitted judges: `ROTE_MODEL_<ID>` /
  `ROTE_BASE_URL_<ID>` env vars beat the IR defaults on all four
  runtimes, and `signature_spec.base_url` reaches any
  OpenAI-compatible endpoint. Node.source (provenance) is excluded
  from the pipeline hash on purpose — metadata must not re-version
  in-flight workflows.
- All three graduator drivers: `ClaudeDriver` (`claude -p`),
  `CodexDriver` (`codex exec` — workspace-write sandbox, no env
  scrubbing since Codex login is only overridden by `CODEX_API_KEY`,
  smoke-tested against the real CLI), `AnthropicApiDriver`
- `rote analyze` — the `plan` to `graduate`'s `apply`: runs the
  graduator, then prints a structural report (node-kind breakdown,
  roteness matching `rote eval`, HITL gates, targetable runtimes)
  instead of emitting runtime code. `--json` for machines; `--out`
  keeps the IR for a later `rote emit`.
- MCP requirements manifest: `Pipeline.required_mcp_servers` (derived,
  never stored — same rationale as `requires_durable_execution`) feeds
  an `mcp_servers` entry — `{server, nodes, tools, auth}` — in
  `analyze`/`emit`/`graduate` output (human, `--json`, and the
  progress-file summary line), joined against the local registry +
  token store. Deliberately **advisory-only**: rote never hard-gates
  on auth (park-on-auth is the backstop); it prints `rote mcp login`
  recommendations for servers that would park a run.
- Data-flow threading: nodes declare `inputs:` (param → source
  reference, grammar in `rote.ir.parse_input_ref`) and all three
  adapters — Temporal, Cloudflare, and DBOS — thread real payloads
  through the DAG, with HITL gate resume payloads participating as
  the gate's result — validated empirically in the runtime e2e tests
- Park-on-auth (ALL MCP-capable runtimes — DBOS Python, DBOS-TS,
  Inngest, Cloudflare): MCP-backed steps with a missing/dead
  credential suspend the run durably; `rote mcp login <server>` (or
  `rote mcp release <server>`) releases every parked run across the
  apps recorded in `~/.local/share/rote/apps.json` (written at
  emit/graduate time). Release channels differ per runtime: DBOS =
  discovery + per-workflow send (portable serialization); Inngest =
  one broadcast event per pipeline (events fan out; no discovery
  needed, but NO buffering for unstarted waits — retry-once covers the
  race); Cloudflare = REST blast to every non-terminal instance
  (no broadcast, but events buffer per-instance, so blasting is
  race-free; needs CLOUDFLARE_API_TOKEN/ACCOUNT_ID, local dev uses
  `wrangler … send-event --local`). Retry opt-outs also differ:
  should_retry (DBOS-py) / shouldRetry (DBOS-TS) / NonRetriableError
  (Inngest) / NonRetryableError (Cloudflare — and its waitForEvent
  THROWS on a 24h-default timeout, hence the explicit "30 days").
  Proven live per runtime: `tests/test_mcp_park_e2e.py`,
  `test_mcp_park_ts_e2e.py`, `test_mcp_park_inngest_e2e.py`,
  `test_mcp_park_cf_e2e.py` (the last also runs the TS MCP SDK live
  inside workerd).
- `rote run` — one-off local execution of either side, built as a thin
  detection/dispatch layer (`rote.runners`) over the proven trial
  primitives: a skill dir runs via `rote.eval.baseline.run_baseline_trial`
  (registry MCP injection + read-only gate), an emitted dir via
  `rote.eval.empirical.run_pipeline_trial` (python in-process; dbos with
  cross-process `DBOSClient` gate delivery — payloads are collected up
  front, safe because DBOS notifications persist per-topic; cloudflare
  under `wrangler dev --local` via `rote.runners.cloudflare`, driving
  the emitted `src/index.ts` router with parked-then-send gate delivery
  in topological gate order — parking is inferred from
  `__LOCAL_DEV_STEP_OUTPUTS` stability because local-dev status lies,
  see the Cloudflare gotcha; inngest via `rote.runners.inngest` —
  emitted serve entrypoint + managed `inngest-cli dev`, forced PUT
  registration, and gate events RE-BROADCAST every poll tick because
  Inngest drops events with no active waitForEvent, with output read
  from the dev server's GraphQL RunComplete op; dbos-ts via
  `rote.runners.dbos_ts` — the emitted main.ts shares dbos-py's
  one-shot CLI contract, signals go Python→TS over the portable
  serialization channel, and an unreachable Postgres falls back to a
  throwaway Docker container; NOTE the TS SDK prints its startup
  banner to STDOUT ahead of the result JSON, hence the trailing-JSON
  extraction; temporal via `rote.runners.temporal` — no subprocess
  dance at all: temporalio's WorkflowEnvironment.start_local() manages
  a real dev server, the worker runs in-process on the emitted
  workflow.py/activities.py with UnsandboxedWorkflowRunner, and
  signals send up front because Temporal buffers them server-side).
  ALL SIX emitted runtimes are now locally runnable.
  Detection understands both the `graduate --out` layout
  (`runtime/<target>/`) and bare emitted dirs (marker files — see
  `rote.runners._detect_runtime`).
- `rote deploy` — push-deploy wrappers (`rote.deploy`): cloudflare via
  `npx wrangler deploy` (whoami-preflight surfaces the session account
  first — ambient wrangler state on this machine is the WORK account),
  dbos/dbos-ts via `npx dbos-cloud app deploy` (verified current: no
  `app register` step anymore; CI auth = `login --with-refresh-token`).
  Temporal/Inngest/python have NO push model — the command prints
  current hosting guidance with doc URLs instead of pretending
  (Temporal Cloud hosts only the server; Inngest syncs via
  authenticated POST to api.inngest.com/v2/apps/<id>/syncs, the old
  unauthenticated PUT-to-your-endpoint flow is deprecated). The
  `rote-cloud` target (`rote.deploy_rote_cloud`) bundles the emitted
  `src/workflow.ts` with `npx esbuild` (esm/neutral, worker-first
  conditions, `cloudflare:workers` + `node:*` external — must match the
  platform's reference bundler) and POSTs the manifest-derived
  DeployPayload to `/v1/pipelines` with a tenant bearer token; the
  emitted cloudflare `manifest.json` is REQUIRED (no TS-regex
  fallback). Proven live against the platform running under
  `vite dev`.
- `rote login` / `logout` / `whoami` (`rote.cloud_auth`) — OAuth 2.0
  device flow (RFC 8628) against the platform's better-auth
  `deviceAuthorization` plugin. BOTH UXes ride the one grant path:
  browser available → auto-open `verification_uri_complete` (code
  pre-filled, one Approve click); `--device`/headless → printed code +
  URL. There is deliberately NO localhost-callback listener — no second
  server-side mechanism to secure. The polled session token is traded
  at `POST /v1/cli/keys` for a durable tenant `rote_…` API key; the CLI
  never persists a session token. Credential at
  `~/.local/share/rote/cloud.json` (0600, atomic;
  `ROTE_CLOUD_CRED_PATH` test override — conftest sets it suite-wide).
  Resolution everywhere: flag > `ROTE_CLOUD_URL`/`ROTE_CLOUD_TOKEN` >
  stored login. `logout` revokes server-side (`DELETE /v1/cli/keys`,
  the key revokes itself) but ALWAYS clears the local store; `whoami`
  verifies live via `GET /v1/me`.
- Login-aware graduate default: logged in + no explicit `--runtime` →
  emit `cloudflare` and auto-deploy to rote cloud after emission
  (`--no-deploy` or explicit `--runtime` opt out; a deploy failure
  exits 1 but keeps all local artifacts and prints the
  `rote deploy … --target rote-cloud` retry). Logged out → today's
  `dbos` default with a one-line hint, never a prompt (CI-safe). The
  parser's `--runtime` default is now `None` — resolution happens at
  the top of `_cmd_graduate`; don't reintroduce a static default in
  the parser.
- `rote register` + `rote serve` (graduated pipelines as MCP tools,
  FastMCP 3.x, stdio + Streamable HTTP — see
  [`docs/mcp-trigger.md`](docs/mcp-trigger.md))
- `rote eval` + the auto-emitted `graduated/scorecard.md` (static
  before/after estimate of speed, cost, determinism; live-fetched
  prices — see the eval gotchas below)
- `rote eval --run` (empirical mode: real trials of both sides —
  `claude -p` for the skill, the emitted python/DBOS app for the
  pipeline with cross-process gate signaling via `DBOSClient`; real
  judge usage captured through the emitted `$ROTE_USAGE_LOG` hook;
  measurements appended to `~/.local/share/rote/eval-corpus.jsonl`)
- 722 tests (698 fast + 24 slow). Run with `pytest tests/` (fast
  only — what runs by default). Slow tests cover the runtime e2e
  suites (Temporal, Cloudflare, DBOS, DBOS-TS, Inngest,
  MCP-over-stdio); the TS ones require a Node toolchain, DBOS-TS
  needs Docker, and Inngest downloads the `inngest-cli` binary. Run
  them with `pytest tests/ -m slow`.

If you find something in the "working" column that doesn't work,
file it as a bug. If you find something in the "stubbed" column
that frustrates you, that's the roadmap — pick it up.

---

## What this file is not

- **Not a README.** The [README.md](README.md) is the public-facing
  front door; it explains what `rote` is to someone landing on the
  repo cold.
- **Not CONTRIBUTING.md.** The [CONTRIBUTING.md](CONTRIBUTING.md) is
  the human-facing collaboration guide.
- **Not design doc.** The canonical design record is
  [`docs/agent-runtime.md`](docs/agent-runtime.md), which captures
  the driver-layer decisions with research citations. Read it if
  you're touching the driver layer.
- **Not the rubric.** The graduator's rubric lives in
  [`skills/rote-graduate/references/`](skills/rote-graduate/references/) —
  that's what the agent reads. This CLAUDE.md is what *you* read.

If information in this file contradicts information in `ir.py`, the
test suite, or `scripts/sanity-check.sh`, **those are the source of
truth and this file is out of date.** Please fix it.
