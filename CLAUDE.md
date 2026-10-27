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
`references/`) and compiles it into a runnable background pipeline:

1. An LLM agent (itself defined as a skill at
   [`skills/rote-compile/`](skills/rote-compile/)) reads the
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

### Layer 1 — The compiler agent (the "brain")

- **Lives at:** [`skills/rote-compile/`](skills/rote-compile/)
- **Consists of:** a `SKILL.md` plus five rubric files
  (`node-kinds.md`, `crystallization-heuristics.md`, `ir-schema.md`,
  `llm-judge-extraction.md`, `implementation.md`)
- **What it does:** classifies every step of a source skill into
  one of 5 node kinds, extracts deterministic procedures into
  Python modules, designs typed signatures for LLM-judge steps, and
  assembles the whole thing into a `pipeline.yaml`
- **Runs under:** a pluggable driver layer (see Layer 2)
- **System-1 is the default compile path.** Before the agent runs,
  [`src/rote/compiler/system1/`](src/rote/compiler/system1/) decides
  the whole structure (kinds, data flow, fan_out, mandatory, gates,
  constants), and the agent gets the smaller
  `skills/rote-compile/SYSTEM1.md` rubric and writes only text. rote
  restores every decision after each agent stop. The full rubric above
  runs under `--no-system1` and for `--update`. `Compiler(system1=None)`
  means exactly that rule, and rote-cloud constructs `Compiler` with the
  default, so flipping it there changes cloud compiles too. Who decides
  kinds and gates is the layered `decider` setting: `rules` (built-in, no
  call), or the `clef` / `jev` decision models through one Workers AI
  client (`system1/decision_model.py`), each with thresholds fitted on the
  dev skills and frozen before a held-out run. Steps still below 0.7 after
  that go to Haiku in one schema-locked call (`system1/tiebreak.py`,
  layered `tiebreak` setting, default `none` because it did not beat
  rules-only fidelity). It bills through the
  compile's own lane via `structured_call` in the inference helper. Don't
  give it its own client or a temperature. See
  [`docs/design/system1-compile.md`](docs/design/system1-compile.md).
- **The skeleton comes from the structure pass by default**
  (`system1/structure.py`; the layered `structure` setting, `--structure`
  > `ROTE_STRUCTURE` > `rote.yaml` > user config > `haiku`, flag rejected
  in cloud mode like `--tiebreak`): one schema-locked Haiku
  call reads the whole skill and returns every node, and code checks each
  one against hard evidence only (declared tool schemas, verbatim quotes and
  constants, sample keys, the IR's loop rules), sending failures back for up
  to two repairs. Keep the checks to evidence: a check that encodes a guess
  about meaning belongs in the prompt, and a word list never does. What
  still fails is resolved conservatively and listed as an Open question,
  never shipped as decided. It and the tie-break run on the compile
  driver's own client and nothing else (`tiebreak.lane_for`): claude -p,
  the api driver's Anthropic client, the openai-api driver's OpenAI client
  with the compile's model, or `codex exec --output-schema`
  (`system1/lane_call.py`, strict `json_schema` on the Responses API). Never add a fallback to an ambient
  `ANTHROPIC_*` key or a `claude` on PATH: it billed a user's subscription
  for a compile they sent to GPT-5.5, and rote-cloud's container has
  neither, so every openai-api compile failed there. A lane with no client
  is refused for the same reason. Every call goes through
  `lane_call.structured_call`, which retries a transient failure (408, 429,
  5xx, a dropped connection) up to three calls on the same lane and model
  and then raises the real error; keep it a retry, never a switch to
  another model or provider. Any other driver fails naming
  `--structure rules`, the opt-out that keeps the rules above. Tests that run
  a compile with a fake driver pass `system1_structure="rules"`, because the
  fake has no lane. Tune its prompt on the dev skills only; the holdout is
  for measuring.
- **A gate must block something** (`structure._check_gate_waiters`). A
  `hitl_gate` that no later node reaches by `reads` or `after` goes back for
  repair naming the gate and asking which step the approval guards; after
  the repairs it is an Open question, never a guessed edge. The one
  exemption is the BDR gold's shape: a gate that is the run's only exit is
  its final sign-off. It is a structure-pass check, not IR validation,
  because every committed BDR pipeline ends in such a gate, and which step
  a gate guards is a reading of the skill. An audit of 15 default compiles
  found 3 of 16 gates blocking nothing.
- **A gate binds what the person decides on, and only that**
  (`structure._check_gate_shows`). The pass names each value a gate shows
  (`<node>.<key>`, a keyless result whole, or `pipeline.input.<field>`) and
  a short message; code checks each resolves to a real earlier output key
  and that every `{name}` is shown, then renders them as the gate's
  `inputs:` and `notify.message_template` (no channel or target: the skill
  names none). Reads it does not show become ordering edges. Before this,
  every gate bound `request` plus whole upstream results, so the compiled
  skill's HUMAN block showed expense-audit's whole input (3.9 KB) and an
  eval's agent spent 5-9 turns finding the review queue in it.
- **The decider double-checks the structure pass, never overrides it**
  (`system1/checker.py`). Its confident disagreements go back to the pass as
  one more repair round; what the pass keeps is an Open question. Keep it a
  second opinion: an override would let a one-step reader beat the model
  that read the whole skill. The default (`auto`) is the first of
  `decision_model.CHECKER_ORDER` with credentials (Clef via
  `ROTE_CLEF_ACCOUNT_ID`/`ROTE_CLEF_API_TOKEN`, then Jev via
  `TYPESAFE_API_KEY`), else no check; change the order there, citing the
  bench that moved it. Never read `CLOUDFLARE_*` for a decision model: deploy
  credentials must not opt a compile into inference (the user's rule).
- **System-1's text is written in parallel** when the driver has
  `generate()` (`claude`, `api`): one schema-locked call per node's
  interface entry, then one per `pure_function` (module + tests) and per
  judge (prompt), all at once under `--max-parallel`, each checked by code
  and re-asked alone ([`src/rote/compiler/system1/generate.py`](src/rote/compiler/system1/generate.py),
  rubric `skills/rote-compile/GENERATE.md`). `codex` and `openai-api` keep
  the single SYSTEM1.md session. See
  [`docs/design/parallel-generation.md`](docs/design/parallel-generation.md).

**When you edit this layer:** you are changing how the agent
behaves. Changes are testable by running `rote compile` end-to-end
against `examples/bdr-outreach/skill` and diffing the output
against the previously committed snapshot in
`examples/bdr-outreach/runs/`.

### Layer 2 — The driver layer (the "runtime for the agent")

- **Lives at:** [`src/rote/compiler/drivers/`](src/rote/compiler/drivers/)
- **Protocol:** every driver implements `CompilerDriver` with
  `name`, `is_available() -> (bool, str)`, and
  `async run(skill_dir, compiler_skill_dir, work_dir) -> DriverResult`
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
- **Seven adapters today:**
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
    the IR timeout explicitly. Targets DBOS 3 (Python) / Transact v5
    (TS); see the DBOS 3 queue gotcha below.
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
  - `SkillAdapter`
    ([`src/rote/adapters/skill.py`](src/rote/adapters/skill.py),
    registered as `skill`) — emits a compiled skill, not workflow code:
    SKILL.md (the source skill's name and description, a short body),
    README.md, and `scripts/` with `plan.json` (`rote.plan.build_plan`,
    live), the verbatim driver (`rote.skill_target._driver`) and helpers,
    and `extracted/`. One manifest covers every file, SKILL.md included.
    The identity comes from a SKILL.md rote can vouch for (`--skill`, or
    `source_skill` checked against `provenance.json`'s digest), never a
    guess. Three tiers: the driver runs what it can in code, escalates
    model work to the agent (`DECIDE`, `LOOP`) and reaches a person only
    through it (`HUMAN`). An external_call with no code (its impl a stub,
    or no impl and no binding the backend calls; a symbol rote cannot
    read, such as a re-export, is shipped code and stays a call) is a `do` step in the
    plan and a `DO` block: the agent does it with its own tools, as the
    source skill did, and answers (`answered_by: agent`, `--said`
    refused, checked by the one schema validator). Which ones is decided
    from the modules the skill ships (`rote.contracts.impl_status`), so a
    compile's dry emit (`allow_stubs=True`) and the real one build the same
    plan; before this, a vendor stub passed verification and failed the
    real emit after a paid compile. A pure_function stub is still refused
    at emit (a compile defect, not agent work), as are PlanErrors. A DO
    block grants nothing: the agent's own client approves its tool calls,
    writes included, so `--allow-writes` (consent for the driver's calls,
    which bypass that approval) neither covers nor gates it. The IR marks
    no step as a write, so every DO block says what a write is and to
    follow the client's approval. See
    [`docs/design/compiled-skill-target.md`](docs/design/compiled-skill-target.md).
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
tested in isolation) but `Pipeline.execution_waves()` (in `rote.ir`;
the adapters' `_execution_waves` just calls it) filters them out of
the top-level DAG, otherwise the workflow would dispatch them twice.

### Valid IR is emittable IR: reachability lives in `rote.ir`

A node's `inputs:` may only name a node in an earlier wave, never a
same-wave sibling and never a `loop_body` sub-node (it has no top-level
result). That rule used to be checked only inside each adapter at emit
time, after `rote compile` had verified the deliverable as clean: a real
bdr-outreach compile paid $4.46 for an IR whose `email_template_loop`
bound its own body sub-node `personalize_email` as an input, then exited
1 at emit. It is now `Pipeline._validate_input_reachability`, so the
compile's repair loop sends it back like any validation error, and the
six adapters' copies are gone. Cycles are a validation error for the
same reason. Loop-body sub-nodes' own `inputs:` are exempt (they
describe, they do not bind).

A `hitl_gate` shows the person its `inputs:` and nothing else (the
compiled skill's HUMAN block is the one place a gate's data is shown), so
`Node._validate_gate_shows_what_it_binds` requires every `{name}` in
`notify.message_template` and every key its `input:` documents to be
bound. Every committed compile that broke this had documented the data
and bound none of it, or printed counts (`{vetted_count}`) no node
computes, and an eval agent spent 5 to 9 turns hunting for the missing
review queue. A gate that only collects data binds nothing. That is why
the rule keys on what the gate says it shows, not on its edges.

For the rules no validator states (python refusing a `hitl_gate`, an
MCP-incapable adapter refusing an mcp-only node, an emitter crash),
verification dry-emits the deliverable for the compile's runtime
(`verify.EmitTarget` / `emit_findings`, wired from `--runtime`), and an
`emit_refused` finding goes back to the agent. Don't add an emit-time
check to an adapter when it can be IR validation; when it can't, the
dry emit already covers it.

### Driver factories accept `**kwargs`

The driver registry's factories have signature `**kwargs ->
CompilerDriver` because `Compiler(model=...)` needs to pass
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

### Sync Python runtimes run async work on ONE background loop

python and dbos steps are plain functions, so every coroutine they need
(an `async def` extracted function, a dbos MCP step, a legacy async
judge) goes through the emitted `_run_async`, which submits it to one
daemon-thread event loop per process. Do not "simplify" it back to
`asyncio.run`: that makes and closes a loop per call, so a module-level
`httpx.AsyncClient` fails from its second call on ("Event loop is
closed"), which is every fan_out element and every retry; and it raises
inside a running loop, where `run_pipeline` lands when async code
imports it. `_call_impl` comes in two forms from one place
(`_py_common.call_impl_helper`, `is_async=True` for temporal), and both
check the *result* for awaitability, so a plain wrapper returning a
coroutine works on every runtime (`tests/test_call_impl_helper.py`).

### A missing MCP endpoint is configuration: never retried, never parked

`resolve_url` / `resolveUrl` raise `RoteMcpConfigError` (Python and all
three TS helpers, detected by `name` in TS like the auth error) when no
endpoint resolves. Every runtime's retry opt-out excludes it next to the
auth error, so a step fails on its first attempt with a message naming
`ROTE_MCP_<SERVER>_URL`. Agent loops reach the same gap through tool
discovery, so `bindAgentTools` / `_mcp_config_for` raise the config error
when it explains a missing tool, and loop steps carry a name-based opt-out
(by name so `main` never imports MCP code, invariant #3). It must not join
the park path: a parked run
waits for a login, and no login supplies an endpoint. Before this, DBOS
spent 4 attempts and surfaced only `DBOSMaxStepRetriesExceeded`
(`tests/test_mcp_config_error_e2e.py`). On Cloudflare local dev the
instance status then reads only `WorkflowFatalError`; the real message is
in workerd's console output, which `rote.runners.cloudflare` reads back
from `wrangler-dev.log`.

### Never return fastmcp's `result.data` from an MCP step

`_result_payload` in `rote.mcp._runtime_helper` must NOT be
"simplified" to `return result.data`. Re-verified on fastmcp 4.1.1
against real TS-SDK servers (v1 and v2) and five hosted servers, it
fails both ways:

- **None for text-only results.** fastmcp hydrates `.data` only from
  structured content. FastMCP servers auto-generate output schemas
  (which is why FastMCP fixtures never caught this), but real-world
  servers, the TS SDK in particular, return JSON as text and `.data` is
  silently None. Found live on the first real-server run: every MCP
  step returned None.
- **Unpicklable for typed results.** With an output schema, `.data` is
  a generated `types.Root` class: DBOS cannot pickle it, and keys that
  are not identifiers come back renamed (`display-name` →
  `display_name`). Found live against `docs.mcp.cloudflare.com`.

The payload is `structured_content` (the wire value, with fastmcp's
`{"result": ...}` wrapper removed only when the server marked it), else
the parsed text blocks. Two things to keep: the marker is
`_meta.fastmcp.wrap_result` on current fastmcp but only the
`x-fastmcp-wrap-result` output-schema flag on 2.x/3.0 servers, so the
schema is still needed; and the TS helpers apply the same rule through
`_FASTMCP_UNWRAP_TS` (one source spliced into the Node and Workers
helpers). The text fallback is shared the same way: `_MCP_TEXT_PAYLOAD_TS`
(spliced into the Node, Workers and binding helpers) applies
`_result_payload`'s exact text rules, which are every text block parsed,
non-JSON kept verbatim, several blocks as a list, and none as null. The
TS helpers used to read only the first block, throw on non-JSON text and
return `{}` for no text, so the same MCP step returned different values
per runtime (`tests/test_mcp_text_payload_parity_e2e.py` runs both
helpers against one real server). The helper's explicit `list_tools()` is not an extra round
trip, because the MCP SDK's `call_tool` lists tools itself on a fresh
session to validate the result (measured, and pinned by
`test_one_tools_list_per_step`). Regressions: `tests/test_mcp_client.py::
test_result_payload_is_plain_json_from_a_real_fastmcp_server` and
`tests/test_mcp_ts_server_e2e.py` (real TS-SDK servers, slow).

### fastmcp 4 hides HTTP status codes; the helper hooks the transport

MCP SDK v2 turns any non-2xx answer into a bare `MCPError` (-32603,
"Server returned an error response") with no cause chain, and the HTTP
stack is httpx2, not httpx. So a 401 can no longer be found by walking
the exception for `httpx.HTTPStatusError`; that check silently matched
nothing and park-on-auth stopped working. `_HttpStatusWatch` builds the
transport's httpx2 client with a response hook and remembers whether
the latest answer was a 401. Regression: `tests/test_mcp_client.py::
test_call_mcp_tool_maps_a_real_401_to_auth_needed` (a real localhost
401, with a 500 as the negative half). Any new `except httpx.` around a
fastmcp call is dead code for the same reason.

Two smaller fastmcp 4 traps: `client.ping()` raises "Method not found"
on the sessionless 2026-07-28 era, which `Client(url)` negotiates by
default (use an authenticated `list_tools()` to trigger OAuth), and an
SDK v2 `ServerSession` is per request, so anything that remembers a
client across requests must hold the `Connection` instead (see
`rote serve`'s list_changed delivery, which also has to publish on a
`subscriptions/listen` bus for modern clients).

### MCP 2026-07-28: what keeps both eras and every registration mode working

Eight things, each with a test that fails when it is undone
(`tests/test_mcp_era_matrix_e2e.py`, `tests/test_mcp_oauth_registration_e2e.py`,
`tests/test_mcp_park_e2e.py`). See [`docs/mcp-client.md`](docs/mcp-client.md).

- **The TS helpers must pass `versionNegotiation: { mode: "auto" }`.**
  `@modelcontextprotocol/client` 2.x defaults to `legacy`, which fails
  against a modern-only server exactly like v1 (`-32022`). Checking the
  import line proves nothing; the era matrix asserts the era each server
  saw. A 401 is `SdkHttpError.status`; v1's `err.code === 401` reads an
  SdkErrorCode string in v2 and never matches, which silently turns
  park-on-auth into a failure.
- **The CIMD document's redirect URIs and `CALLBACK_PORTS` are one fact.**
  A CIMD server compares redirect URIs by exact string match and no spec
  defines a loopback port wildcard, so the login binds a published port.
  `tests/test_mcp_client.py::test_cimd_document_matches_what_the_login_offers`
  pins `docs/oauth/rote-cli.json` to the code, and the copy served by
  rote-cloud's marketing site must match it too. Add ports to the
  document before the CLI uses them, never after. Selecting a port must
  bind it (`bind_callback_sockets`), and the callback server must run on
  those sockets: a check now and a bind later lost the port to a
  concurrent login in CI after the redirect URI already named it.
- **rote refuses registrations the SDK would send on its own.** Without
  a `registration_endpoint` the SDK POSTs a guessed
  `<resource-origin>/register` (GitHub: not its authorization server).
  `RoteOAuth._refuse_registration` stops that, the emitted helper refuses
  any registration (it parks instead), and an issuer-bound pre-registered
  client that the SDK dropped is not silently replaced by CIMD or DCR.
  The tests assert on the authorization server's hit log, because the
  failure is a request that should never have been sent.
- **Refresh goes to the stored `token_endpoint`.** Before discovery the
  SDK refreshes at `<resource-origin>/token`. For a path issuer or an AS
  on another host that 404s, and an unattended run parks on a credential
  that was fine. `retarget_token_request` is shared by the CLI and the
  emitted helper. FastMCP fixtures cannot catch this: their AS is the
  server's own origin root.
- **Never offer the CIMD document unverified.** The SDK does not fall back
  to DCR when a CIMD server rejects it, so an unpublished or stale
  document breaks logins that DCR serves. `_check_offered_document`
  fetches it once per process when the AS advertises CIMD and withdraws
  the URL unless it equals the packaged `client_metadata.json`.
- **A token file is bound to an endpoint and an issuer.** Files are keyed
  by server name, so `stored_for_url` gates every read (else a repointed
  name gets the old bearer token), and `_with_bound_issuer` fills
  `client_info.issuer` from the file (the SDK treats a missing one as
  matching every issuer).
- **`--no-wait` keeps the verifier in a detached listener, never on disk.**
  `rote mcp login --no-wait` spawns `python -m rote.mcp.pending_login
  <id>` (own session, stdio on `/dev/null`) running the same `login()`;
  it hands the authorization URL back over a pipe and waits for the
  browser. `--complete` reads the 0600 record and waits a bounded time.
  Don't move the PKCE state into the record for a paste-back: a pasted
  callback goes to the listener (`--callback-url`), which is the only
  thing that can redeem it. Inheriting the caller's stdout hangs the
  agent's shell tool; sharing its process group kills the listener at
  the tool's cleanup (`tests/test_mcp_login_no_wait_e2e.py`). The
  callback's `state` is checked before it counts
  (`auth._StateCheckedReady`): a forged request no longer ends a login,
  and the SDK's mismatch error, which quotes the expected state, never
  fires.
- **`RoteMcpInputRequired` is not an auth error.** A server that wants
  input mid-call (`input_required`, or `-32021` for an undeclared
  capability; TS SDK servers send the latter because rote declares no
  capabilities) fails the step on every runtime, and every retry opt-out
  exempts it. Making it a `RoteMcpAuthNeeded` would park the run for 30
  days waiting for a login that answers nothing.

### `mcp.args` literals are values, never code

A real BDR compile wrote fixed tool arguments ('management-levels', 'BDR
do not contact') into `mcp.args`, which held only references, and the IR
rejected them. `{literal: <str|int|float|bool>}` (`rote.ir.MCPLiteral`)
is the fixed-value form. It is a mapping so it can never be confused with
a reference: a bare string is always a payload key or a constant name.
Strict types, so `"3"` stays a string. No `null` (omit the argument) and
no lists or objects (a constant, named by reference, already carries
those).

`mcp_tool_arguments` returns `(arg, "literal", value)` and every emitter
renders it exactly as it renders a constant: `json.dumps` in TS,
`_py_literal` in Python (invariant #7). Two traps the malicious-string
parity test (`tests/test_mcp_args_literal_parity*.py`, the four MCP
runtimes) found:

- `json.dumps` escapes a character outside the BMP as a surrogate pair,
  which JS reads correctly and Python reads as two lone surrogates. Python
  string literals go through `py_str_literal`: raw non-ASCII, but every
  character in Unicode category Cc, Cf, Zl, Zp or Cs escaped
  (`escape_invisible`). Raw, a bidi override, zero-width space or BOM is
  the Trojan Source shape (CVE-2021-42574), and a lone surrogate (PyYAML
  makes one from `"\ud800"`) cannot be written as UTF-8 at all. The same
  rule covers docstrings (`safe_docstring_line`), TS comments
  (`safe_block_comment_line`) and the Cloudflare README; TS literals are
  ASCII `json.dumps`, already safe. `_pipeline_hash` falls back to an
  escaped encoding only when the JSON encoder refuses a lone surrogate, so
  no existing workflow hash changes. Regression:
  `tests/test_emitted_source_is_visible.py`.
- `JSON.stringify` leaves those three raw, so the DBOS-TS runner's
  trailing-JSON parser splits on `\n` only, or a valid result is cut
  apart.

python and temporal have no MCP backend and refuse an MCP-bound node
without an `impl`, so `mcp.args` never reaches them.

### A compiled contract is decided by data, not by the model's reading of prose

A skill run as an agent adapts to whatever input arrives. A compiled
pipeline has fixed field names. Measured with `rote bench --fidelity`:
every one of 36 expense-audit compiles failed on the skill's own input,
because the skill describes it in prose and each compile invented names
(`expense_lines`, `lines`, `reporting_period`), invented required inputs
the skill never takes (`fx_rates`, though the skill reads
references/fx-rates.md; `controller`), and indexed rows by
`line["expense_id"]` where the data says `id`.

`rote.compiler.grounding` decides the contract with code wherever data
exists: a sample input (`--sample-input` > `input.json` in the skill dir >
the input a baseline ran on) is reduced to a schema by
`rote.probe.infer_schema` and handed to the agent as an `INPUT
CONTRACT`. Verification then checks it, and `rote.compiler.prefix` runs
the pipeline's pure functions (and MCP nodes a baseline recorded, matched
on exact arguments) on the sample in the generated-test sandbox. A
`KeyError` there goes back to the agent with the file and line. MCP tool
arguments are grounded the same way, in each tool's own input schema:
a live `tools/list` of the registered servers the skill names, else the
skill's bundled snapshot (`mcp-tools.json` in the skill dir,
`rote.tool_schema.load_bundled_tools`). Live always wins per server; a
live schema that differs from the snapshot for a bound tool is reported
as a stale snapshot, and the summary and compile-report.md name what
grounded each server. Things to preserve:

- **The prefix runner is a dumb interpreter.** `_prefix_runner.py` is
  stdlib-only and copied into the sandbox scratch dir. The semantics
  (payload from `inputs:`, omitted optional fields left out, constants
  merged last, `fan_out_element_param`, `mcp_tool_arguments`) live in
  `prefix.build_plan`, which reuses the adapters' own helpers. If you
  change how an emitted step builds its payload, the plan must follow,
  or the prefix checks a different program than the one that runs.
- **Never guess a tool result.** An MCP node with no recorded call, or
  a recorded call with other arguments, ends the prefix on its branch
  and becomes an open question (`contract_questions`), not a fixture.
- **Generated tests cannot catch this.** The model that wrote
  `line["expense_id"]` writes tests with `expense_id` rows. The sample is
  the only input in the loop the agent did not write.
- **Tool argument names come from the tool, not the skill.**
  `rote.compiler.tool_grounding` lists the tools (`tools/list`, read-only,
  through the emitted pipelines' own non-interactive client, all servers
  concurrently) of the registered servers the skill names, before the
  agent runs, plus the servers the user has logged in to. A logged-in
  server whose tool the skill calls (a code span or a call, never a
  product name in prose: Atlassian's own skills say "Jira" and call
  `getJiraIssue(...)` but never name `atlassian`) grounds the compile
  directly. Otherwise its tools are *offered* to the structure pass, with
  each tool's own description (`safe_description`: one escaped line, cut
  at 200 characters, labeled as data), and the pass may bind a step to
  one ("use read-only Jira search" became `searchJiraIssuesUsingJql`).
  Code checks the binding like any declared tool, and only a server the
  plan binds becomes grounding (`adopt_offered`). Descriptions go to the
  structure pass only, never to the files the agent reads. Only those: a
  third-party skill must not trigger authenticated requests, and silent
  token refreshes, against servers the user has not approved
  (`--tool-discovery all` opts in, `none` opts out on any driver). Schemas
  are server-supplied text, so the agent sees only
  identifier-shaped names and JSON type names, labeled as data.
  Verification checks each `mcp:` binding's arguments against a declared
  schema; one that is not a flat object schema (top-level `anyOf`,
  non-local `$ref`, no `properties`) is an open question, never a
  failure. The schema
  is read one way everywhere (`rote.tool_schema`): the bench's fixture
  servers refuse exactly what verification refuses, and the bench
  registers those fixtures for each compile, so compile and fidelity see
  one schema. A schema inferred from a baseline's recorded calls never
  fails a compile: it only shows what someone sent.

### `--backend api` is a compile mode, not only an emit flag

With `--backend api` every external_call runs its `impl:` module, so the
agent must write real vendor calls, and a call from training-data memory
fails at run time. The Compiler owns this (`rote.compiler.api_docs`), on
both the legacy and the System-1 path:

- **The backend reaches the Compiler as `emit_target.external_backend`.**
  The Compiler, not the CLI, grants the driver `web_tools` and adds the
  brief to the prompt, so any caller that compiles for `api` gets both.
  Claude: WebSearch/WebFetch. api: the Messages API's server-side
  `web_search_20250305` / `web_fetch_20250910`, and NOT the newer
  versions: those provision code execution, and a real compile spent
  about $14 of extra input tokens running code in it. Resend a
  `pause_turn` as-is; it is not a stop. Codex: `-c web_search="live"`
  plus `--json` (`codex exec` has no `--search`). openai-api: no web
  tool exists on chat/completions, so it reports itself unavailable
  rather than compiling from memory.
- **Citations are checked, and fetches are the evidence.** Each vendor
  function carries `# docs: <url>` (or `# docs: skill` for an internal API
  the skill itself documents). Verification sends back an implemented
  external_call that cites none, and, when the driver reports fetches, a
  cited page the agent never fetched. The report marks each cited page as
  fetched in this compile or not. The fetched list comes from the drivers'
  `tool` events (their `url` / `query` fields), so a driver that does not
  put them there makes the report say "unknown".
- **TypeScript runtimes are refused at compile.** They emit a stub for
  every external_call, so `compile --backend api --runtime cloudflare`
  (including the logged-in default) exits 2 before the agent runs. The
  Compiler refuses it too, for callers that skip the CLI (the cloud
  container, the bench).
- **The backend is recorded beside the IR, never in it.** A compile writes
  `backend.json` next to `pipeline.yaml` (`rote.compiler.backend`).
  `rote emit` and `compile --update` use it when `--backend` is not
  passed, and warn when the flag contradicts it, because emitting the
  other backend drops code the compile wrote for this one. python and
  temporal never call a binding, so they advertise no MCP server for one.

### Newer Anthropic web tool versions start a code execution container

`web_search_20260209`+ and `web_fetch_20260209`+ turn on "dynamic
filtering": the API provisions code execution for the request on its own,
with no tool in your `tools` array asking for it. Inside the api driver's
compile loop the model used that container as a second workspace, 70+
`code_execution` / `bash_code_execution` calls inside two requests, billed
as 3.69M and 2.91M input tokens ($8.04 + $6.18, measured on the Cloudflare
AI Gateway logs). The same skill compiles for about $1 on the claude driver.

So the api driver pins `web_search_20250305` / `web_fetch_20250910` with
`max_uses` caps and `max_content_tokens`, refuses to send any other server
tool (`_check_server_tools`), stops the run when a response carries a
server tool it did not ask for, and stops when one request bills more than
`DEFAULT_MAX_INPUT_TOKENS_PER_REQUEST` input tokens. Do not "upgrade" the
versions without a measured run. The ceiling is checked on the response,
so it stops the next request, not the one that crossed it.

### Fidelity for `--backend api` pipelines fakes the vendor, from its spec

`rote bench --fidelity` runs an api pipeline against a local fake of each
vendor API (`rote.vendor_specs.fake`), built from the vendor's spec and
answered from the case's `http:` fixtures (`rote.bench.http_fixtures`).
Things that will bite if you touch it:

- **Routing is a `sitecustomize`** (`rote/vendor_specs/intercept/`), put on
  the pipeline process's `PYTHONPATH`. It patches httpx, requests,
  httplib2, urllib and `google.auth.default`, rewrites the faked hosts to
  the fake, and BLOCKS every other host of a vendor in play
  (`ROTE_HTTP_FIXTURE_BLOCK`), so a bench run cannot reach a real API. It
  must stay standard-library-only at module level: it runs in the app's
  environment, not rote's.
- **A fixture is held to the spec at load** (`rote.vendor_specs.fixtures.
  validate`): a field the response schema lacks is an error, so a fixture
  can never teach a pipeline a field the real API does not return.
- **`reads: <server>.<tool>[i]` is what makes API and MCP reads
  comparable.** Without it an API read keys by operation and counts as
  unexpected against the skill's reference.
- **Fixture sets are stored keyed by `<vendor>/<api>/<version>/<name>.json`**
  (`rote.vendor_specs.fixture_store`, `ROTE_FIXTURE_DIR`), validated and
  stamped with the spec revision on save, re-validated on load. Only
  authored data goes there: a live check compares shapes and never keeps
  what it read.
- **`rote run --live-check` never runs the deliverable's code.** A live
  run of generated code holds the user's real credentials, and no patch
  over its HTTP clients can promise it only reads: review6's reproducers
  wrote through an unextracted urllib call, a helper module, a host alias
  and http.client while the intercept guarded the checked call. So
  `rote.vendor_specs.live` builds each entry read itself from the static
  extraction (the spec-resolved operation, the URL template, the
  parameters and headers as `Expr` templates filled from the node's inputs
  and the user's environment, Google calls as the client library would send
  them with the application-default token) and sends only GET/HEAD through
  one urllib client with redirects off. A read it cannot build exactly is
  "not live-checked", never sent. A docs operation named like a write is
  refused whatever its method. Three more rules, from review6's exfil
  reproducer (an agent-written docs param plus `os.environ["AWS_SECRET_..."]`
  sent the secret): trust comes from the vendor or the user, never the
  agent. A credential is attached only from a vendor-published spec
  (Google's OAuth token, which the check supplies itself, gated on the
  spec's provenance, `source="google-discovery"` from rote's store, and a
  googleapis.com host, never on the agent-writable `vendor`; a docs spec
  claiming a vendor rote reads a spec for is refused at compile time); a docs spec's
  `auth` is the agent's choice of secret, so it is withheld unless the
  user passes both `--allow-host <host>` and `--allow-credential <ENV>`
  (`--yes` never covers either), and any other `{$env:...}` refuses the
  whole check. A docs-spec host is called at all only when it equals, or
  shares the registrable domain of (`rote.vendor_specs.psl`: the bundled,
  pinned Public Suffix List with its private section, so `a.github.io` and
  `b.github.io` are two owners; docs on another registrable domain, such
  as docs.slack.dev for slack.com, need `--allow-host`), a cited page in
  `web-research.json`
  (rote's own record, written by the Compiler over any file the agent
  wrote), or `--allow-host` names it; "every cited page was fetched"
  vouches for nothing. Every request is built first and the plan is
  confirmed (`--yes`, or a TTY answer; neither exits 2 having sent
  nothing). The report holds the operation's path
  template and schema paths (a map's keys and any non-field-like live key
  are `{key}`), and an error is its exception type and HTTP status, never
  its text (a URL in it carries query credentials). For a docs spec, extra
  live fields are not drift; for a vendor's own spec they are.
- **Vendor libraries are installed per requirement set** into
  `<spec cache>/../bench-deps/<hash>` with `uv pip install --target`, from
  a curated import-to-distribution map (`DISTRIBUTIONS`). An unmapped
  import is an error, never a guessed package name.

### Vendor requests are checked against the vendor's spec, in code

On `--backend api`, verification reads every request an external_call
module makes (`rote.vendor_specs.extract`: httpx/requests calls with URL
templates rebuilt from literals and constants, and Google client-library
`build(...)` chains, including helpers that wrap `build` and resources
bound to names) and checks it against the vendor's machine-readable spec
(`rote.vendor_specs.check`): operation, parameter names, types, enums,
required parameters, body fields, and the response fields the code reads.
Violations are errors that go back to the agent. Three rules:

- **The spec comes from the vendor, never from memory.** Google: the
  Discovery directory (`https://discovery.googleapis.com/discovery/v1/apis`)
  names each API version's document. Specs are cached under
  `ROTE_SPEC_CACHE_DIR` (default `$XDG_CACHE_HOME/rote/specs`, else `~/.cache/rote/specs`) as
  `<vendor>/<api>/<version>.json`, the `ApiSpec` model, which is also the
  shareable format. An expired cache is refetched; a failed refetch is
  `SpecUnavailable`, never a stale read.
- **What cannot be checked is listed, not failed.** A URL built at run
  time, a host with no spec source yet, or a spec service that is down
  shows in the report as "not checked: <why>". An unreachable spec must
  never fail a paid compile.
- **OpenAPI comes from a curated registry of vendor-published documents**
  (`rote.vendor_specs.openapi.REGISTRY`, host -> URL), never from a
  community directory: APIs.guru's Slack entry is an archived 2021 spec.
  A vendor's own spec outranks a docs spec the compile wrote for the same
  host (`check.spec_for_key`, the one precedence rule for the check, the
  live check and the bench), and a docs spec naming a registered vendor is
  refused outright (`docs.PUBLISHED_VENDORS`). Add a vendor by fetching and parsing its
  document first, and register every host its `servers` name at any level
  (Stripe uploads at files.stripe.com, GitHub at uploads.github.com).
- **A false error is worse than a missed one**, because the agent rewrites
  correct code to satisfy it. So a part of a request the spec does not
  describe is `partial` ("no violations; not checked: <what>"), never
  "ok" and never a finding: a field under an object that lists no
  fields, `additionalProperties: true`, an `anyOf` branch that is open, a
  `$ref` rote cannot follow, a body or response with no JSON schema. An
  object that lists its fields is closed (the vendors leave
  `additionalProperties` out), and a field is missing only when no branch
  of an `anyOf` / `oneOf` has it. The correct-call corpus
  (`tests/fixtures/vendor_specs/correct_calls/`) must stay at zero
  findings; add a vendor's real calls there with its fixture.
- **Send parameters in the spec's own style.** Stripe's 282 `expand` and
  63 `created` parameters are `style: deepObject`: `expand[]=...`,
  `created[gte]=...` (the spec has no bare `expand=`), and its form bodies use
  the same brackets (`metadata[order_id]`). GitHub's `x-multi-segment`
  path parameters become `{+name}` (a file path keeps its slashes). The
  body's media type is checked too: Stripe takes form bodies, never JSON.
- **GraphQL is checked by graphql-core against the vendor's schema**
  (`rote.vendor_specs.graphql.REGISTRY`, endpoint -> SDL file or the
  endpoint's own introspection), plus the variables sent (names, and a
  literal value's type) and the response fields read against the
  document's selection, inline and named fragments followed. Four traps:
  - **Build with `assume_valid=True`.** GitHub's public schema fails
    graphql-core's schema validation (a deprecated field implementing one
    its interface does not deprecate), and `validate()` raises a
    `TypeError` on such a schema. The request is still validated in full.
  - **The operation is chosen as GraphQL's GetOperation does**: one
    operation, or the one `operationName` names; several and no name is a
    real request error; a computed `operationName` is not checked.
  - **A document is read only from a name bound once.** A name rebound,
    or bound under an `if`, could send either document (a mutation where
    a query was read), so it is "built at run time", never guessed. The
    same goes for a body dict.
  - **A read below a custom scalar is not checked** (AniList's `Json`);
    below a built-in scalar or an enum it is a finding.
  A GraphQL read is a POST: the live check sends only a document whose one
  operation is a query, to a declared endpoint. A GET to an endpoint is not
  checked.
- **Where no spec exists, the agent writes one, from fetched pages.**
  `vendor_specs/<vendor>.<api>.<version>.json` (`rote.vendor_specs.docs`)
  becomes an `ApiSpec` with `source="docs"`, checked the same way and
  labeled lower trust everywhere. Each of its sources must be a page
  fetched in this compile. A host with no spec of either kind is a
  `spec_missing` error on the api backend, never a silent "not checked".
- **Tests never fetch a spec.** conftest replaces `http_get_json` and
  points `ROTE_SPEC_CACHE_DIR` at the test's tmp dir. Tests use the trimmed
  real documents in `tests/fixtures/vendor_specs/` through an injected fetch.

### A tool result has one shape: what the step receives

Four places read an MCP result: the emitted helper at run time, and at
compile time the baseline's schema inference (`rote.probe`), the prefix
run's recorded stubs (`rote.compiler.prefix`) and System-1's interface
(`Context.known_result_schema`). All four go through
`_runtime_helper.returned_value` / `returned_schema`, so a fastmcp tool
that returns a string (wrapped on the wire as `{"result": ...}` under an
`x-fastmcp-wrap-result` output schema, DeepWiki's shape) is a string
everywhere. A recording is decoded before its schema is inferred; without
that, every text-recorded result was typed `string`. When nothing is
recorded, the outputSchema the server declares is the shape, never a
guess from the skill's words: a guessed `url` where Microsoft Learn says
`contentUrl` made every live run raise. The compile writes the declared
schemas to `compiled/declared-tools.json` itself, after the agent's work
dir is moved, because the post-compile enrichment and cross-check run in
the CLI and must not read an agent-written file.

`tool_grounding.result_schema` is that one rule (recorded, then declared),
shared by System-1's `Context.known_result_schema` and verification's
`result_field_checks` (#253). An `inputs:` read of `<mcp node>.output.<field>`
is an error (`result_field_missing`) when the known result is a string or a
list, or a closed object without the field, and an Open question when
nothing is known. An object that admits other keys proves nothing.

The same rule holds at the other boundaries a generated test cannot see,
because the model that wrote the module wrote its fixtures too (#291).
`rote.compiler.boundaries` reads each module for literal keys
(`p["k"]`, `p.get("k")`, `p.pop("k")`) on the parameter bound to a gate's
answer or the pipeline input, and refuses one the other side does not
declare. `tool_grounding.unsent_constant_findings` refuses an MCP node whose
constants `mcp.args` never names (`adapters._common.unsent_constants`): an
error when the tool's schema names its parameters, else an unresolved call
that blocks the run (`NodeImplementation.blocks_run`) and asks for the
server's registration. Never guess the parameter names. `mcp.args: {}` on a
node with no constants stays a zero-argument call.

### An mcp-bound node is a BARE tool call — post-processing must split

The default backend emits an mcp-bound step as exactly "call the tool,
return the result". If a skill step is "call X, then filter/match/
reshape", the transformation half must be its own `pure_function` node
or it silently vanishes at emission (live failure: match-by-name logic
lost, workflow crashed dereferencing a declared-but-never-computed
key). The rubric (`node-kinds.md`, external_call) mandates the split;
`rote.probe.cross_check` flags violations as `output_mismatch` (node
declares `output` keys the observed tool result can't provide) and
`rote compile` prints them as WARNINGs.

### `temperature` is rejected outright by current Claude models

A judge carrying `temperature` 400s on every call against a current
Anthropic model (`invalid_request_error: \`temperature\` is not
supported on this model`). The compiler used to write
`temperature: 0.0` on judges as a determinism instinct, which made
every compiled pipeline dead on arrival the moment `ROTE_MODEL_<ID>`
pointed at a current model — found live, first real gateway run.

Two halves of the fix, keep both: the rubric
(`llm-judge-extraction.md`) says never set `temperature` — determinism
comes from schema-locked decoding (`output_config.format` /
`json_schema`), not the sampler. When the IR *does* set one, emitted judges
render it as a `ROTE_TEMPERATURE_<NODE_ID>` env knob whose empty value
drops the parameter entirely (`_sampling_kwargs()` in Python,
`...(TEMPERATURE.trim() ? … : {})` in TS). Don't "simplify" it back to
a literal keyword argument: a baked constant cannot be escaped without
a re-emit, which defeats the model-override knob next to it.

The SDKs dropped sampling too, in two steps, and the Python helper
handles both in `_anthropic_sampling_options`. anthropic 1.0 removed
`temperature` / `top_p` / `top_k` from `messages.create`'s signature, so
`**sampling` raised `TypeError` before any request on every judge whose
IR set a temperature (the floor, 1.15, included). anthropic 2.0 moved to
API version 2026-10-21, which 400s on any of the three even in the body.
So sampling rides in `extra_body` with `anthropic-version: 2023-06-01`
pinned on that request only, the route Anthropic's migration guide gives.
A judge with no sampling sends the SDK's request untouched. Measured
live: the 2023-06-01 pin works with `output_config`, and a non-default
temperature applies only to models older than 4.7 (claude-haiku-4-5 and
claude-sonnet-4-6 accept it). claude-sonnet-5-5 answers "`temperature` is
not supported on this model" and claude-haiku-5-5 "`temperature` is
deprecated for this model." (`TEMPERATURE_REJECTIONS` in
`tests/_messages_api.py`). The TS
emitter still spreads `temperature` into the body: that holds only while
`ANTHROPIC_SDK_NPM_VERSION` stays below `@anthropic-ai/sdk` 1.0 (same
change, same API version). Regression, the floor and the newest release
against a stub that refuses sampling under 2026-10-21:
`tests/test_judge_sampling_sdk.py`.

### Not every skill should become a compiled skill

A compiled skill stops for the agent at each judge and agent loop, and every
stop is a round trip that re-reads the conversation. Measured 2026-10-26 on
Sonnet 4.6: expense-audit (no model stops) ran 3.6x faster and 62% cheaper
compiled; Atlassian's search-company-knowledge (4 judges) ran 4x slower and
3x costlier, with the same substance. So a `--runtime skill` compile checks
its plan right after the structure pass (cents) and stops before generation
when it has more than `rote.compiler.worth.MAX_MODEL_STOPS` (2) judges and
loops. Gates are not counted (the original asks the person too), nor
`--judges helper` judges (API calls, not round trips), nor loop-body
sub-nodes. `--compile-anyway` overrides it. `rote try` and `rote setup`
compile through it, so they inherit the check. Workflow runtimes are not
checked: there a judge is an API call. Move the line only with a measured
eval behind it.

### First-run output: three contracts that live in one place each

Found by a fresh-install run of the published wheel, where every one of
these read fine to someone who already knew the answer:

- **Install hints come from `rote.extras.install_hint`.** The PyPI
  distribution is `rote-cli`; `rote` is somebody else's package, so a
  hand-typed `pip install rote[api]` installs the wrong thing.
  `tests/test_first_run_output.py` fails on the bare form, and on a
  hand-typed `rote-cli[` outside `rote/extras.py`.
- **A failed subprocess reports its tail** (`rote.eval.empirical.failure_excerpt`).
  The exception is at the end; a `[:800]` head slice showed a newcomer
  DBOS's startup banner cut off mid-word and nothing else.
- **`--json` always puts one object on stdout.** `main()` wraps every
  `--json` handler; a nonzero exit with empty stdout gets
  `{"ok": false, "exit_code", "error"}` built from the `error:` line the
  handler printed. New error paths only need the usual
  `print(f"error: ...", file=sys.stderr)`.

`rote run` also refuses stubs up front (`rote.runners.preflight_pipeline`,
reading the emit manifest via `rote.adapters._common.unimplemented_stubs`),
so a stub never burns a durable runtime's retry budget before failing.
A stub is a function raising with `STUB_SENTINEL` (`[rote stub]`, in
every stub rote emits) or, in Python, one whose whole body is `raise
NotImplementedError` (the shape compile agents write), at any depth,
methods included, `@abstractmethod` excepted. The marker counts only
where it is executable: in Python as a `raise` argument (AST), in TS
after `strip_ts_comments`. A raw-text match kept a filled-in module that
merely mentions the marker in a comment listed, and `rote run` refused
it. Never go back to a `NotImplementedError` substring match either: a
real module that raises it on an unsupported branch was refused with no
way past it. `--allow-stubs` is the escape hatch.

### Current Sonnet rejects forced `tool_choice`

`claude-sonnet-5-5` answers every request with `tool_choice` type
`"tool"` or `"any"` with HTTP 400: `tool_choice: type "tool" and "any"
are not supported for this model.` (verified live 2026-10-20;
claude-haiku-5-5 and claude-sonnet-4-6 still accepted it, and the
Cloudflare AI Gateway `/ai/v1/messages` route passes the same 400
through). Every emitted judge used to force a tool call, so pointing
`ROTE_MODEL_<ID>` at the current Sonnet killed every judge call: the
same dead-on-arrival class as the `temperature` gotcha above.

Schema-locked decoding is now `output_config: {format: {type:
"json_schema", schema}}` on every Anthropic lane, for every model, with
no branch on model ids: the Python helper's `_call_anthropic`, the TS
judge emitter, and both agent loops' one extra turn (with `tool_choice:
none`, because the history holds `tool_use` blocks). Three things to
keep:

- **The request schema is rewritten, in one place.**
  `structured_output_schema` in `rote.inference._runtime_helper` closes
  every object (`additionalProperties` omitted or `true` is a 400),
  turns `oneOf` into `anyOf`, and moves bounds, `pattern`, `maxItems`,
  `minItems` above 1 and unknown `format`s into the description (all
  400s as written). The judge's own validator, built from the original
  schema, still enforces them. Python runs it per request; TS runs the
  same function at emit time. `tests/_messages_api.py` is an
  independent copy of the measured rules, and the Messages stubs refuse
  what it refuses, so the suite fails where the API would.
- **Two shapes cannot be expressed, and are never approximated.** An
  object that admits undeclared keys (no declared properties, or
  `additionalProperties` set to `true` or a schema, as in a fixed key
  plus a typed map) and a value with no type. Only an object that *omits*
  `additionalProperties` is closed. Closing a map would make the model
  drop its keys with nothing noticing.
  An anthropic judge with either fails at `rote emit`, naming the path.
  An agent loop's output often has one (an IR type name the IR never
  defines), so its extra turn falls back to asking for `submit_result`.
  That fallback keys on `STRUCTURED_INEXPRESSIBLE` in the message, so any
  other normalizer error still propagates.
- **The SDK floor is load-bearing.** `anthropic>=1.15` (has
  `output_config`) reaches every Python runtime's install instructions
  through `_py_common.vendor_sdk_requirements`; an older SDK fails every
  judge with an unexpected-keyword error.
- **Only `end_turn` is an answer.** `refusal` and `max_tokens` raise a
  plain RuntimeError naming the stop reason; a truncated JSON prefix
  must never reach a downstream node.

The `claude-cli` lane (`claude -p --json-schema`) and the OpenAI lane
are separate mechanisms and were unaffected (the CLI handles
claude-sonnet-5-5 itself).

Effort rides in the same `output_config`. Unset, the API runs `high`.
`ROTE_EFFORT_<NODE_ID>` > `ROTE_JUDGE_EFFORT` set it for judges and agent
loops on every lane, and an empty value omits it (the temperature knob's
shape). On the `claude-cli` lane it is `claude -p --effort`, added by one
function (`_cli_effort_flags`) to both the judge and the agent-loop
command lines. The CLI does not fail on an unknown level: it warns on
stderr and runs at its default, exit 0. So before any paid call the
helper asks the CLI itself with `claude --effort <level> --version`
(free, ~50ms, validated before `--version` exits), once per process:
first with a level nothing accepts, which must draw a warning rote
recognizes (any line naming effort and saying unknown / invalid /
unsupported), then with the real level, which must not. Never hardcode
the level list in the helper; it goes stale. If a CLI release rewords
the warning past recognition, the self-check fails loudly instead of
letting every bad level through. Tested against a stub CLI
(`tests/test_cli_effort_check.py`). The default stays unset on
data: on 2026-10-20, 8 example judges x 6 inputs, `low` cut output tokens
15% (sonnet-4-6) and 33% (sonnet-5-5) but changed decisions the default
made the same way twice, including dropping an ARR stated in the input.
Don't name the pipeline-wide knob `ROTE_EFFORT`: that is the compiler's,
and it would silently retune every judge a developer runs. On the Node
TS runtimes a judge only sees the env vars `judge_env_arg` threads, so a
new knob must be added there too (the temperature knob was missing until
this was found).

### Vendor SDK errors do not survive a durable step boundary

`anthropic.APIError` / `openai.APIError` subclasses take keyword-only
`response`/`body` arguments, so `BaseException.__reduce__`'s
`(cls, args)` cannot rebuild them. A runtime that persists a failed
step by pickling the exception — DBOS does — therefore replaces the
real status and body with `TypeError: APIStatusError.__init__()
missing 2 required keyword-only arguments` at the join. That masked
the temperature 400 above completely.

Emitted judges wrap the SDK call and re-raise through
`_durable_error()`, a plain `RuntimeError` carrying type, HTTP status,
model, node id, and the original message (regression:
`tests/test_dbos_adapter.py::test_judge_translates_sdk_errors_so_they_survive_a_durable_step`,
which asserts the translated error actually pickle-round-trips). This
bites hardest on the fan_out enqueue path, where the failure is
observed in a *different* process from where it was raised — any
durability boundary is also an error-fidelity boundary.

### The TS tool runner's `done()` deadlocks; use `runUntilDone()`

`client.beta.messages.toolRunner(...)` returns an async iterator plus two
ways to wait on it, and they are not interchangeable. `done()` *observes*
a loop somebody else drives — its own docs say it "works even if the async
iterator hasn't yet started, and will wait for an instance to start" — so
a single-consumer caller that never iterates hangs forever, silently, with
no output and no error. `runUntilDone()` consumes the iterator itself.
The Python twin's `until_done()` has no such split, which is exactly why
this is easy to port wrong. Found by running it (rc=124, zero output);
tsc and the SDK's types are both happy with the deadlocking version.

Related, same file: report the turns that actually ran, not the cap.
The count is assistant messages
(`runner.params.messages.filter(m => m.role === "assistant").length`),
`number | null`, and goes to stderr (TS) or the usage log's `iterations`
field (Python), since a loop's result is now its declared output. The
Workers AI lane reports null because `runWithTools` genuinely does not
report it. Echoing back `maxIterations` made every loop look saturated. The Python
`_run_agent_via_sdk` has no public `params` accessor, so it drives the
runner's iterator itself and counts each yielded message, summing usage
over every turn. Then `until_done()` on the drained runner returns the
final turn without another request.

### An agent_loop's tools carry no server — `tool_servers` supplies it

`Node.tools` are bare names (`zoominfo_search_contacts`): one loop may
reach several servers, and an `MCPBinding` is a single server/tool pair.
That gap was not cosmetic. `Pipeline.required_mcp_servers` derives from
server *names*, so a pipeline whose only MCP usage is an agent loop
reported **zero** required servers — `analyze`/`emit`/`compile` and the
`rote mcp login` advisories all said there was nothing to authenticate.
BDR is exactly that shape, so the flagship example demonstrated the bug.

`Node.tool_servers` (tool name → server, agent_loop only, keys validated
⊆ `tools`) closes it. Three things now depend on it:

- **The advisories are correct.** `required_mcp_servers` counts both
  declarations — a node's `mcp:` binding and a loop's `tool_servers`. BDR
  reports five servers instead of none.
- **A resolved tool binds from its own server and no other.** Two servers
  exporting one tool name is precisely what an allowlist cannot
  disambiguate; first-wins discovery would pick whichever answered first.
  Python narrows its `--allowedTools` cross product the same way.
- **Workers works at all.** There is no registry in workerd —
  `resolveUrl(env, server, …)` reads env bindings — so the server name
  *must* be known at emit time.

Three sources feed the emitted loop, in descending authority: the IR's
`tool_servers` (knowledge), other nodes' `mcp:` bindings (a guess — the
loop probably reaches the servers the rest of the pipeline does), and at
run time the local registry plus `ROTE_MCP_SERVERS` (the escape hatch).
Only the first is trustworthy, which is why only it feeds the advisories.
A partial map is valid and an unresolved tool still works via discovery;
the emitted module names what it could not resolve rather than failing
later with a bare "no server provides this tool".

Two behaviors to preserve:

- **Nothing guesses.** `rote.probe.enrich_pipeline` writes `tool_servers`
  from what the baseline actually *called* (`cross_check`'s
  `resolved_agent_tools`), and leaves a tool served by two servers in one
  trial unresolved — pinning either would silently send the loop to the
  wrong endpoint. An authored value always beats inference. The compile
  pins the same way from tool grounding, on every path, after the agent's
  last stop (`tool_grounding.loop_tool_pins`, written as a line edit so
  the agent's YAML comments survive): one grounded server offering the
  tool pins it, two leave it unpinned as an Open question. A loop tool no
  grounded server offers is `loop_tool_unserved`, only when a live
  tools/list answered for every server the skill names (no listing proves
  nothing). It is a review gap, never an error: rote does not contact a
  registered server the skill never names, which may offer the tool. The
  structure pass's own `<loop id>_tool` placeholder is exempt by that exact
  name.
- **Discovery is best-effort, binding is not.** A server that is down may
  not be one this loop needed, so it is recorded and skipped; a declared
  tool no reachable server provides is fatal. When an auth failure
  plausibly explains the gap, that failure is rethrown so the workflow
  parks and `rote mcp login` releases it, instead of dying with a
  misleading "unknown tool".

### `rote run` must see a park, and only the runner can release a local one

A parked run used to look like a hang under `rote run`. The run-side
watcher (`rote.mcp.parked`) reads each runtime's park signal: the DBOS
`rote_auth_status` event (now `{awaiting, node}`), and on Inngest and
Cloudflare a `rote-auth-park {json}` line the emitted park prints before
its `waitForEvent`, because neither local dev server reports what a run
waits on. Traps, all found by running it:

- **The prompt goes to stderr.** `input()` writes its prompt to stdout,
  which is `rote run`'s result JSON. On a pty with stdout piped the user
  is asked a question they never see and the run waits forever.
  `_stderr_input` exists for this; don't "simplify" it back to `input`.
- **Replays reprint the marker.** Both TS runtimes re-execute workflow
  code after a release, so the line prints again. Dedupe on its `wait`
  field (the wait's step id), never on server or node.
- **`rote mcp login` cannot reach a `rote run` dev server.** The
  registry-wide release targets Inngest's default port and the
  Cloudflare REST API. So the watcher releases its own run when the
  server's token file or registry entry changes. On DBOS it waits a few
  seconds first, because login already released registered DBOS apps.
- **A login writes the token file twice.** Client registration lands
  first with no token. Releasing on that write wakes the run into the
  same failure, it parks again, and the finished login then looks like
  "no change". `credentials_settled` waits for an access token.
- **A Worker never sees the local token store.** On Cloudflare the
  watcher shows the export-to-`.dev.vars` fix and neither offers a login
  nor self-releases, since a login there would "succeed" and change
  nothing the Worker can read.
- **An auth park looks like a gate park to the Cloudflare stability
  heuristic.** While the watcher reports an active park, the runner must
  not spend a gate payload on it.
- Non-TTY never prompts and never opens a browser. The browser login is
  offered only when stdin and stderr are both terminals.

### An agent loop returns its declared output, through a submit tool

Downstream nodes index an agent loop's declared `output` keys
(`process_invoices_loop.output.invoice_results`), but the loop helper
returned `{result, provider, iterations}` on every runtime, so none of the
keys existed. Python raised KeyError in the next step, Cloudflare carried
`undefined` on silently (a measured run reported `"undefined / NaN"`), and
DBOS-TS and Inngest failed tsc with TS7053. No test had a downstream node
reading a loop key.

The contract now, from `rote.adapters._common.agent_loop_output` (one
derivation, all six runtimes):

- **A declared output is submitted, not parsed from prose.** The loop gets
  a `submit_result` tool whose input schema is the declared output (keys
  required, `additionalProperties: false`). The CLI lane uses
  `--json-schema` instead, the same forced structured output a judge
  uses. A single-type output is submitted as `{result: ...}` and returned
  bare. A loop with no `output` still returns `{result: <final text>}`.
- **Submitting does not end the loop.** It runs until the model ends its
  turn, and the last submission wins. A loop that ends (or hits its cap)
  without one gets ONE more request, then fails. That request is
  schema-locked the judge way (`tool_choice: none` plus
  `output_config.format`, never a forced `tool_choice`; see the gotcha
  on the current Sonnet) when the output schema fits structured outputs,
  and otherwise only asks the model to call `submit_result`. At the cap the runner
  has already run that turn's tools, so the nudge joins the trailing
  user message rather than re-running anything (the Python runner does
  not append a normally-ended final turn; the TS one does, hence the
  check on both sides). The helper builds the assistant message param by
  hand: `Message.to_param()` exists only in newer SDKs, and an emitted app
  runs on whatever version it installed.
- **Every lane validates the same way.** The submit tool is `strict: true`
  (Anthropic's grammar-constrained tool input, no beta header) when the
  schema fits that subset: every leaf typed, every object closed, and at
  most 16 `anyOf` / type-array parameters (each nullable field is one;
  past that the request is a 400). A named type is description-only, so
  its loop stays non-strict. Either way, every lane checks the answer
  with rote's one schema validator, and every failure reads `agent loop
  <id> (<provider>) did not hand back its declared output: <reason>`.
  Workers AI cannot force a final call, and says so.
- **One validator, one Python source, one TS source, on every lane.**
  Judges and agent loops alike enforce the full schema through it.
  `schema_errors` in `rote.inference._runtime_helper` checks every answer
  rote does not get from a decoder that enforces the whole schema: every
  judge on every Python runtime (in `call_judge`, after the lane returns
  and before the caller's Pydantic model, which drops bounds too), agent
  loops (the submit tool and the final `_declared_output` check), and a
  compiled skill's judge, gate, input and tool-argument checks (the driver
  loads it as `_rote_inference.py`, shipped in every compiled skill).
  `project` beside it mirrors the Pydantic `model_dump`: the driver hands
  it downstream, and `_declared_output` returns it, so `3.0` under
  `integer` comes back `3` as on TS. The rules are the judge's Pydantic
  model read strictly (no coercion, `3.0` is an integer, an optional key
  may be null) plus what that model drops: bounds, `multipleOf`, lengths
  in code points, `pattern`, item counts, `uniqueItems`, `const`, `allOf`,
  typed map values. Its TS twin is `SCHEMA_ERRORS_TS` in `_ts_common.py`,
  emitted once per app as `signatures/_roteSchema.ts` beside any judge or
  loop (`needs_schema_helper`). Every TS judge checks its raw answer with it
  before Zod, and `_roteInference.ts` imports it. Where the answer goes:
  - A judge that breaks the schema fails the call naming each path (`judge
    <id> answered outside its output schema: output.grade: 11 is above the
    maximum 10`), and the step's own retry asks again. A judge has no
    writes, so a rerun is safe.
  - An agent loop's submit tool refuses a bad submission back to the model
    as the tool result (`Not recorded: ...`, the same text in both
    languages), so the model fixes it inside the loop. Failing the node
    instead would make a durable retry rerun the whole loop, writes
    included. The final check stays as the backstop for the CLI lane and
    the forced turn.
  - A `pattern` an engine cannot compile is refused by name in both
    languages. `rote.ir.check_schema_patterns` refuses one Python cannot
    compile at IR load, on every judge, node and input schema. JavaScript's
    compilability cannot be checked there without node, so a pattern only
    Python accepts (`(?P<name>...)`, inline `(?i)`) fails on TS at the
    first answer. The engines also disagree on `\d` / `\w` over non-ASCII
    text.
  Change a rule in both languages, then run
  `tests/test_schema_validator_parity.py -m "slow or not slow"`. It runs
  every example IR's judge, loop and gate schemas, plus a positive and a
  negative case per rule, through Python, the TS source under real node,
  and (for judges) the emitted Pydantic models.
- **Types:** scalars, lists and `X | null` map exactly, and
  `output_schema` (evidence) beats the type names. A named type the IR
  does not define (`InvoiceResult`) accepts any JSON value and, on TS, is
  an alias in the user-editable `src/types.ts` (`export type InvoiceResult
  = JsonValue`). The loop's return type, and so every downstream read,
  uses the alias, so a typed reader compiles once the user narrows it.
  EmitWriter keeps the edit. Cloudflare types the parked loop's result as
  `Awaited<ReturnType<typeof loop>>`, not `Record<string, unknown>`, or the
  alias would be erased at the read. `rote emit` prints one `note:` line
  per unchecked type.
- **Metadata leaves the result.** Provider and turns go to the usage log
  (Python) or stderr (TS). Putting them next to the declared keys would
  break a single-type output and pollute any whole-output consumer.
- **TS `JsonValue` is spelled out five levels deep, not recursive.**
  Cloudflare's `step.do` checks results against `Rpc.Serializable`, which
  fails TS2589 ("excessively deep") on a recursive type.
- **Python agent loops honor `ROTE_BASE_URL_<ID>`** like the TS ones and
  the judges. The Python loop used to ignore it, so a test pointing it at
  a stub reached the real API.

Regression: `tests/test_agent_loop_output_parity*.py` runs all six
runtimes live against a stub Messages endpoint, with a downstream node
reading two declared keys, and typechecks invoice-push on the three TS
runtimes. All of it fails on the old helper.

### A Cloudflare agent_loop is parkable, so it leaves its parallel wave

Its tools are MCP tools, so `bindAgentTools` and every in-loop tool call
raise the same auth signal a bound step does — an agent loop therefore
gets the park-on-auth wrapper on Cloudflare (invariant #3's park-on-auth
would otherwise not hold on the logged-in default runtime). Two specifics:

- The release event is derived from the failure at run time
  (`authEventType(err, fallback)` parses the server out of the message)
  because one loop may reach several servers and `ROTE_MCP_SERVERS` can
  add more. Own properties do not reliably survive the `NonRetryableError`
  re-throw; the message does.
- Parkable steps stay OUT of `Promise.all` (see the parallel-wave gotcha),
  so an agent loop no longer runs concurrently with its wave siblings.
  That is a real throughput cost on the slowest node kind, accepted
  because `waitForEvent` inside a promise combinator is undocumented and
  its timeout *throws*, which would reject every sibling.

### Emitted judges do not contain the vendor call

A judge module holds its Pydantic models, prompt, schema and operator
knobs; the call itself goes through `call_judge` in
`signatures/_rote_inference.py` — the verbatim source of
`rote.inference._runtime_helper` (never hand-edit the emitted copy; fix
the module — byte-equality test in `tests/test_inference.py`). That
helper resolves *who pays*: `claude-cli` (subscription, local only),
`api` (the user's key), `rote-cloud` (tenant account), selected by
`ROTE_INFERENCE_<NODE_ID>` > `ROTE_INFERENCE` > auto-detect
(cheapest-to-the-user first). See [`docs/inference-runtime.md`](docs/inference-runtime.md).

Traps if you touch this:

- **The provider is not in the IR and must never be.** A
  `pipeline.yaml` is shared, committed, and sometimes third-party — a
  billing decision encoded there travels to people it doesn't belong
  to. `signature_spec.client` stays (wire format is behavior); the
  provider is a deployment fact. Consequence to preserve: the same
  emitted artifact runs on a subscription locally and a key in CI with
  no re-emit and no diff.
- **`build_subscription_env()` lives in the helper**, not beside the
  Claude driver, and `drivers/claude.py` / `eval/*` import it back. Two
  copies is exactly how the `ANTHROPIC_API_KEY` scrub gets fixed in one
  place and silently drifts back to key billing in the other.
- **The refusal is at `rote deploy`, not `rote emit`.** Emit cannot
  know where the directory is headed — that's the point of the bullet
  above. Don't "move it earlier".
- **An explicit endpoint rules out `claude-cli` in auto-detect.** The
  proven AI Gateway config is `ROTE_BASE_URL_<ID>` + a bearer token; a
  laptop with `claude` on PATH would otherwise ignore it silently.
- **The helper is stdlib-only at module level** (vendor SDKs import
  lazily inside functions) and imports nothing from rote. That
  constraint is what makes the verbatim copy safe.
- **No custom exception classes in the helper.** A custom class only
  unpickles where its module is importable, and the observer of a
  failed durable step isn't always the process that raised it — the
  same failure mode as the vendor-SDK gotcha below. Plain
  `RuntimeError` carrying type / status / model / node / provider.

### `--json-schema` is a tool, and tools lead the prompt-cache prefix

The parallel System-1 path makes many `claude -p --json-schema` calls that
share one system prompt (GENERATE.md plus the skill). The schema becomes
the forced StructuredOutput tool, and the cache prefix is tools, then
system, then messages, so two calls with different schemas share nothing:
measured, seven pieces with their own schemas each wrote ~9k cache tokens
and read none. Two rules keep the prefix shared, and both are pinned in
`tests/test_system1_generate.py`:

- **Every call uses `ANSWER_SCHEMA`**, one shape with every field required.
  A call fills the fields its task names and code checks it did. Don't
  give a piece its own schema to "lock it tighter": that re-buys the
  prefix on every call.
- **The first call runs alone** until it finishes or `prime_seconds`
  pass, because a cache entry is readable only once its request's
  response has started. Calls started together all write it.

Related: Haiku 5.5 thinks at length even at effort low (3k to 17k thinking
tokens per piece), so it was slower than Sonnet 5.5 per piece and once
ended a call with `structured_output_retry_exhausted`. A failed call is
asked again once (`CALL_ATTEMPTS`), and a piece that still fails leaves
its module unwritten for verification to report, never the compile dead.

### An MCP result no recording shows: its consumers accept every shape

The interface infers an MCP-bound node's result from the skill's words,
and the guess can be wrong in the one way that fails every run: an
object with a `threads` key for a Gmail tool that returns the list
(ops-report scored 0.50 on that). `Context.unverified_tool_result` marks
such a node, and every consumer's card says to accept the list, an
object wrapping it, or JSON text, and to test each. A recorded baseline
result replaces the guess with an inferred schema.

### `claude -p` gets schema-locked output, and costs 37k tokens if you let it

`claude -p --json-schema <schema> --output-format json` is a **forced
tool call** under the hood (`stop_reason: "tool_use"`) and returns the
validated object in `structured_output` — the subscription lane has the
same structural guarantee as the SDK lane, not "prose that parses".
Prompt goes on **stdin**, not argv (judge prompts carry whole
documents; argv is size-limited and visible in `ps`).

The default invocation ships Claude Code's entire coding-agent system
prompt: measured ~37k cache-creation tokens and 11.5s for a
one-sentence judge. `--system-prompt <one line>` + `--tools ""` +
`--setting-sources ""` was supposed to fix that, and on a machine with
claude.ai connectors it did not: Claude Code 2.1.302 still loaded the
user's connector and plugin MCP schemas (`--tools` does not touch MCP
tools) plus the auto-memory index of the repository the pipeline ran
from. A real "say hi" judge through the helper measured 30.3k tokens,
$0.183 cold; one greeted the developer by name from that memory. The
connectors load asynchronously, so the same call ranged from 1.2k to
28k tokens run to run.

The helper now passes one shared `_CLI_ISOLATION_FLAGS` tuple on BOTH
paths. Each flag was measured by removing it from the full set:

| Removed | Judge tokens (input + cache) |
| --- | --- |
| nothing | 1,250 |
| `--strict-mcp-config` | 30,300 |
| `--setting-sources ""` | 25,600 |
| `--settings '{"autoMemoryEnabled": false}'` | 3,070 (from a repo with memory) |

`--setting-sources ""` does not cover auto memory; only the setting
does (or `CLAUDE_CODE_DISABLE_AUTO_MEMORY=1`, rejected because
`build_subscription_env` is shared with the compiler). `--bare` would
do all of it but refuses OAuth, which is the whole lane; `--safe-mode`
measured the same 1.25k but is documented as a troubleshooting mode
whose scope may change, so the explicit flags win.

The agent loop takes the same tuple plus its own `--mcp-config`:
strict mode means it sees only the servers rote resolved. It also now
passes `--tools ""` when it has servers, because `--tools` removes only
built-ins: before, the loop carried 31 coding tools (Bash, Write,
Agent...) at ~31k tokens a turn. A real two-turn loop against a local
fastmcp server went from 120k tokens / $0.38 to 2.8k / $0.012 with the
tool still bound and called. Removing any flag silently restores its
overhead. Also: the CLI reports API failures *inside a zero exit*
(`is_error: true` in the envelope), so returncode alone calls a failed
judge a success.

The compiler's `ClaudeDriver` passes `--strict-mcp-config` too, with no
`--mcp-config`: it wires no MCP (live compile-time MCP is the
in-process drivers' job), and on this machine the user's connectors put
357 tools into the first compile request, past 200k tokens, which the
subscription rejected outright ("Usage credits are required for long
context requests"). Strict, that request is 70.7k. The compiler still
inherits the user's settings, CLAUDE.md, auto memory, output style,
permission mode and 25 unused built-in tools (11k tokens with all of
those isolated too); that is a separate change because it alters what
the compile agent is told, and needs a real compile eval to land.

### A prompt-cached run reports almost nothing in `input_tokens`

`claude -p` always runs prompt-cached, so `input_tokens` holds only the
few tokens that were neither written to nor read from the cache. A
measured one-word reply reported `input_tokens: 3` beside
`cache_creation_input_tokens: 122956`, and `total_cost_usd: 0.7378`.

The compiler driver summed the plain pair only, with a docstring
explaining that cache fields were dropped "to match the api driver". That
reasoning is the bug: the api driver sends no `cache_control` at all, so
two fields are complete for it and catastrophically incomplete here.
Compiling the rote-compile skill against itself reported a 22-minute,
62-turn run as `tokens in=113 out=1919; cost $0.029`.

Three things now hold, each with a regression test verified by breaking
the code (`tests/test_claude_driver.py`, `tests/test_cli.py`,
`tests/test_compiler.py`):

- **All four buckets are counted.** `_usage_delta` returns
  `{input, output, cache_write, cache_read}`, and every display adds the
  cache buckets into the reported input rather than hiding them.
  Claude repeats input/cache usage across content blocks sharing one
  `message.id`; count each ID once. Assistant `output_tokens` is a
  placeholder: show output as pending until final `modelUsage` (whole
  query, including subagents) or `usage` (main loop) arrives. Recovery
  without final usage retains input/cache and marks output incomplete.
- **The cost key is `total_cost_usd`.** There is no `cost_usd` key in the
  result envelope, so the old read returned `None` on every run while the
  estimate sat one key away. Prefer the CLI's whole-query estimate to
  rote's single-model estimate; neither is authoritative billing data.
  When changing this logic, consult the
  [SDK cost-tracking contract](https://code.claude.com/docs/en/agent-sdk/cost-tracking).
- **Cache rates span the whole fetch.** models.dev publishes them per
  model and `_parse_models_dev` already read them, but `build_catalog`
  kept only the three tier representatives'. The compiler's default
  model is not one, so a default compilation had nothing to price its
  cache with. `PricingCatalog.rates_for` is the accessor that covers
  every model; `model_price_for` stays narrow. An older on-disk cache
  has no such key and degrades to "no published rate" rather than
  reading as free.
- **An unpriceable run reports no cost, not a wrong one.** When a run
  touched the cache and the rates are genuinely unknown, `_cost_usd`
  returns `None`. Billing cache reads at the plain input rate would
  overstate them roughly tenfold.
- **A judge's usage row names the model the CLI ran, not the alias.**
  `claude -p --model sonnet` runs `claude-sonnet-5-5`, and only the
  envelope's `modelUsage` says so. The emitted helper logs one row per
  `modelUsage` entry with the CLI's `costUSD` as `cost_usd`, which
  pricing takes as is: the CLI bills a 1h cache write at twice the 5m
  rate a catalog publishes. A row logged under `sonnet` matched no
  catalog entry, so the whole subscription lane went unpriced.

The general lesson, and it generalizes past this file: a usage number is
only as complete as the fields you chose to read, and a field you never
read fails silently and looks like a small number.

### Sonnet is the default, not Opus

Both `ClaudeDriver` and `AnthropicApiDriver` default to
`claude-sonnet-4-6` rather than Opus, because Opus is ~5× more
expensive and Sonnet is fully capable of following the structured
rubric. We learned this the hard way — the first two BDR runs with
Opus exhausted a Claude Max/Pro "extra usage" budget in two
attempts (~$3.50 each). Don't "fix" the default to Opus unless you
have specific evidence that a skill needs it.

### Thinking, not input, is what a compile pays for

Six real compiles profiled from their session transcripts (2026-10-18,
Sonnet 4.6 at Claude Code's default `high` effort): output tokens were
57-67% of the cost, and adaptive thinking was 62-76% of those output
tokens. Cache reads of the rubric were cheap by comparison (about 24k
tokens, read once and cached). Output also sets the wall time: the CLI
streams about 75 tokens per second, so 94k output tokens is about 21
minutes. Worst of all, the agent drafted the whole compile in one
thinking block right after reading the rubric, hit the per-response
output cap with no tool call made, and the CLI threw the block away.

The fix has two halves, keep both: `ClaudeDriver` passes
`--effort low` (`DEFAULT_EFFORT`, pinned by
`tests/test_cost_defaults.py`), and SKILL.md's "Think in files" work
order has the agent write `pipeline.yaml` first and each module with its
test, instead of planning everything in its head.

Effort is a layered setting like `model`: `--effort` on compile and
analyze > `ROTE_EFFORT` > project `rote.yaml` > user config >
`DEFAULT_EFFORT`. The driver scrubs `CLAUDE_CODE_EFFORT_LEVEL` from the
child, because Claude Code ranks that env var above `--effort`: an export
meant for interactive use would silently re-price every compile and make
rote's precedence a lie. Only `low`, `medium` and `high` are accepted.
`max` is the costliest level there is. Measured together: expense-audit went
from $2.25 / 986 s to $1.20-1.31 / 423-513 s, ops-report from
$2.46-3.54 / 17-30 min to $0.87 / 329 s, with bench quality held (every
pure function implemented, tests green, node and kind accuracy at or
above the `high` runs). BDR at `low` (judges, agent loops, gates): the
first pass took 12 turns, $1.96 and 857 s, against 63-74 turns,
$4.46-5.14 and 27-49 min at `high`. Kind accuracy, gates and judge/loop
detection matched or beat the `high` run's. A full BDR rerun at `low`
($2.85, 65 turns, 20 min, against $5.14, 74 turns, 49 min) ended with
four citation "errors" that were verbatim but for an ASCII `--` for the
skill's em dash and `\"` for its inner quotes. The provenance normalizer
now treats both as punctuation, and the same deliverable then verifies
clean (51/51 tests, dbos and cloudflare dry-emit ok). If one regresses, `medium` ($1.91 / 916 s on expense-audit)
is the fallback. To profile a compile, read its
transcript under `~/.claude/projects/<work-dir-slug>/<session_id>.jsonl`:
the `cost-state` records carry `thinkingTokens` per invocation.

### rote writes compile-report.md; the agent does not

The report was the agent's last file, and none of it was a decision:
the kind breakdown, gates and each node's reason are in `pipeline.yaml`,
the crystallization log is the `# skill:` citations, and the Open
questions are the `# rote: assumption:` markers plus the stub messages.
Writing it cost 17 to 66 seconds of output per compile (profiled
2026-10-19), repair rounds rewrote it, and the `unlisted_assumption`
check sent whole compiles back to repair when its question count fell
short of the markers. `rote.compiler.report.render_compile_report`
renders it from the deliverable after the final verification (from
`output_dir`, after an update's merge), and the CLI appends its
`## rote verification` section as before. The Open questions keep the
`###` shape `provenance.open_questions` parses, so `rote bench`'s `OQ`
is now "distinct assumptions + stubs" and comparable across runs. The
`unlisted_assumption` finding is gone with it. Two consequences to keep:
a judgment call a reviewer must see belongs in a marker or a stub
message (the rubric says so), and `provenance.assumptions()` reads markers
in `extracted/`, `signatures/` and `pipeline.yaml` (a YAML comment on a
node, or a judge prompt), so a structural guess has a route into the
report too. Marker text is rendered behind `> `, so a marker reading
`### OQ-9` cannot forge an entry, and every turn now ends in one sentence,
because the closing summary was 400 to 650 output tokens nobody read.
Tests: `tests/test_compile_report.py`, each verified by a mutation.

### The compiler child writes a 5-minute prompt cache

On a subscription within plan usage, Claude Code writes the main
conversation's prompt cache (`-p` runs included) with the 1-hour TTL,
billed at 2x the input rate (5 minutes is 1.25x). A compile's calls
follow each other within seconds, so the hour buys nothing. Profiled
compiles put an estimated 35 to 40 percent of `total_cost_usd` in
1-hour cache writes. Estimated, because `total_cost_usd` is Claude
Code's client-side estimate: the saving is real in it, but nothing shows
it changes how a subscription's plan limits are consumed. `ClaudeDriver`
sets the documented per-bucket controls, `CLAUDE_CODE_PROMPT_CACHE_TTL=5m`
and `CLAUDE_CODE_SUBAGENT_PROMPT_CACHE_TTL=5m` (`PROMPT_CACHE_ENV`,
Claude Code 2.1.242+), on the compiler child only. They outrank the
settings and `ENABLE_PROMPT_CACHING_1H`; only `FORCE_PROMPT_CACHING_5M`,
which the docs frame as a debugging override, ranks higher
(https://code.claude.com/docs/en/prompt-caching#choose-the-ttl-yourself).
Confirm it in a transcript: `usage.cache_creation.ephemeral_5m_input_tokens`
carries the writes and `ephemeral_1h_input_tokens` is 0. Unmeasured: a
single response, or a repair round's verification, that runs past 5
minutes expires the cache and costs one full rewrite (seen once, $0.23).

### Sonnet 5.5 is faster and cheaper, but not yet the default

Measured 2026-10-19 with `rote bench` (one run each, effort `low` unless
named), against Sonnet 4.6 at `low`:

| skill | model | wall | cost | tests | notes |
| --- | --- | --- | --- | --- | --- |
| expense-audit | 4.6 low | 489 s | $1.36 | 91 | |
| expense-audit | 5.5 low | 183 s | $0.65 | 25 | |
| expense-audit | 5.5 medium | 264 s | $0.81 | 42 | |
| expense-audit | 5.5 high | 405 s | $1.12 | 69 | no repair round |
| ops-report | 4.6 low | 427 s | $1.15 | 58 | |
| ops-report | 5.5 low | 246 s | $0.78 | 14 | no pipeline input, empty gate payload |
| ops-report | 5.5 medium | 249 s | $0.77 | 15 | |

Structure held on 5.5 (kind mix, wave order, gates, every compile
verified; the lower `kind`/`nodeR` on 5.5 low are matching artifacts
like `check_duplicates` for `find_duplicates`). What did not hold is
coverage: 5.5 writes about half the code and a third to half the test
assertions (ops-report: 33 asserts against 88), and at `low` it dropped
the pipeline input schema and the gate's resume payload. Fidelity comes
before speed, so the default stays 4.6 until `rote bench` has a
behavioral-equivalence check that can say whether the leaner modules
behave the same. 5.5 `medium` is the candidate to measure then: 46 and
42 percent faster, 40 and 33 percent cheaper. Haiku 5.5 is not one: it
was slower (402 s, 92k output tokens) for $0.27.

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
anything).

**The trap that fix sets:** because `never` casts to anything, a
reference-site cast that is wrong for real types compiles against the
stubs and fails only once the user types an implementation. All three TS
runtimes shipped `"<exit>": <exit>_result as Record<string, unknown>`,
which is TS2352 the moment the node returns `Promise<string>` (found by
an eval on ops-report; `rote run` passed because casts are erased). A
single `as Record<string, unknown>` only compiles from a type that
overlaps a Record: an anonymous object type does, an `interface` (no
implicit index signature), an array, or a scalar does not. So exit
results are returned uncast (`exit_return_ts_lines`) and field access is
plain `<id>_result["field"]` (`ref_to_ts_expr`), both in `_ts_common.py`.
Indexing `never` is legal, so the stubs still compile. Don't "fix" the
field access with `as unknown as Record<...>`: it compiles everywhere
precisely because it turns off tsc's field-name check, so an IR field the
user's type names differently becomes `undefined` at run time instead of
a TS7053.

The same constraint bites judges, whose output type comes from Zod. A
free-form object (`{type: object}` with no properties) and an untyped
field used to be `z.record(z.string(), z.unknown())` and `z.unknown()`,
so a judge returning one failed `step.do` with TS2769 (deal-monitor,
found by `rote bench`). They are `JSON_VALUE` now (`JSON_VALUE_ZOD_TS`, a
`z.custom` over a JSON type spelled out five levels deep, declared in a
signature module only when used), the same non-recursive shape agent-loop
outputs use, because a recursive type fails TS2589 there.

Checking the stubs compile proves nothing here.
`tests/test_ts_typed_impls_e2e.py` replaces them with typed
implementations of every output shape and runs `tsc` on all three
runtimes (positive half), then renames one referenced field and requires
tsc to fail on exactly that (negative half). The negative half is not
optional: on Inngest, `step.run` returns `Jsonify<T>`, an anonymous
object type that overlaps a Record, so a restored `as Record` field cast
passes the positive half there and only the drift case catches it. Run
the suite after touching any TS emitter's typing story.

### A parallel wave is not automatically parallel in the emitted code

`_execution_waves` says which nodes MAY run concurrently; each adapter
decides whether to take it, and the two mistakes here are easy to
reintroduce because neither fails a test that didn't exist yet.

Cloudflare computed waves and then emitted every `step.do` sequentially
— the concurrency survived only as a `// ─── Wave N ───` comment, on the
runtime that is also the logged-in default. It now emits
`await Promise.all([...])`, which is Cloudflare's documented primitive
(the ✅ example in "Rules of Workflows"; their own visualizer has a
`ParallelNode` for it; no documented per-instance cap on concurrent
steps). Three rules to preserve:

- **Never wrap the wave in an outer `step.do`.** That advice is specific
  to `Promise.race`/`any`, where losing branches are discarded. Wrapping
  burns a step, subjects the whole wave to the 1 MiB step-result cap, and
  collapses per-node retry policies into one.
- **`Promise.all`, not `allSettled`** — opposite of DBOS-TS, on purpose.
  Reaching the rejection means that node already exhausted its own
  retries, and every sibling's result is persisted under its own step
  name, so a workflow retry replays survivors from cache. (DBOS-TS uses
  `allSettled` for a different reason: a bare `Promise.all` can crash the
  Node process on unhandled rejections.)
- **HITL gates and MCP-parkable steps stay sequential.** Both suspend on
  `waitForEvent`, whose behavior inside a promise combinator is
  undocumented — and whose timeout *throws*, which would reject the whole
  wave.

Temporal had the mirror-image bug: `_emit_wave_call` emitted
`retry_policy=`, but the `asyncio.gather` branch emitted only the
timeout, so a node silently lost its declared retry budget the moment it
gained a sibling in its wave — and parallel fetch waves are exactly where
the flaky network calls live. If you add a parallel branch to any
adapter, diff it against the single-node branch field by field.

### A Postgres container is "ready" over the socket before it is ready

The `postgres` image's init phase runs a socket-only server, then
restarts it. `pg_isready` over the socket says ready during that phase,
so an app connecting right after it gets "Connection terminated
unexpectedly". Check over TCP (`pg_isready -h 127.0.0.1`), which answers
only once the final server is up. This only showed in CI, where nothing
sat between readiness and the first connection.

It turned into a hang because the emitted DBOS-TS `main()` called
`DBOS.launch()` before its `try`. A launch that fails part way leaves
pools and timers alive, so without the `finally` shutdown the process
never exits and the runner reports a bare timeout. Launch stays inside
the `try` (regression:
`tests/test_dbos_ts_e2e.py::test_a_failed_launch_exits_instead_of_hanging`).

### An impl's closed input type is checked, not cast away

The implement skill tells users to replace a TS stub's `unknown` input
with a real interface, and that broke the build on all three TS runtimes:
DBOS-TS step wrappers took `Record<string, unknown>` (TS2345), and
Cloudflare and Inngest passed `unknown` values (the pipeline input, a
fanned element) straight into the typed parameter (TS2322). The
typed-impls e2e only ever used `input: unknown`, so nothing caught it.

Every call that reaches an impl now asserts each value to the impl's own
declared type for that key (`expr as ImplInput<typeof fn>["key"]`, the
`IMPL_INPUT_TYPE_TS` alias, emitted only when used), and the DBOS-TS
wrapper takes `ImplInput<typeof fn>`. A stub's `unknown` input reads as
an open record, so the scaffold compiles too. The assertion is per value,
never on the object, so drift still fails both ways: a key the impl
declares that the IR never sends (TS2741/TS2345), and a key the IR sends
that the impl does not declare (TS2339 on the lookup). Judges, agent
loops and MCP calls keep the open record. Regression:
`tests/test_ts_typed_impls_e2e.py`, closed-input positive case plus both
negative halves.

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
   options — there's nowhere to put them. When no node declares a
   `retry:`, the budget is 0, not the SDK's default of 3: an absent
   `retry:` means one attempt on every runtime (Cloudflare emits
   `retries: { limit: 0 }` for the same reason; `tests/test_retry_parity*.py`).
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

### DBOS 3 queues exist only after launch; emitted apps start through `launch()`

DBOS 3 (Python) and DBOS Transact v5 (TS) removed direct queue
construction: `Queue("name")` raises `DBOSException` at import, and
`DBOS.register_queue` needs a launched runtime because queues now live
in the system database. A workflow enqueued on a queue nobody registered
does not fail; it sits in `ENQUEUED` forever. So the emitted `main.py`
declares `QUEUE_NAME`, enqueues via `_enqueue` (a thin
`DBOS.enqueue_workflow(QUEUE_NAME, ...)` wrapper), and exposes
`launch()` = `DBOS.launch()` + `DBOS.register_queue(QUEUE_NAME)`. Any
in-process launcher, tests included, must call `main_module.launch()`.
A bare `DBOS.launch()` hangs the first parallel wave, which shows up
only as a test timeout. Two more v5 traps on the TS side: the SDK no
longer reads `dbos-config.yaml` at launch (only the `dbos` CLI does, so
`rote._dbos.dbos_ts_system_database_url` ignores its
`system_database_url` key), and starting or enqueuing a *step* directly
now throws, which is one reason the DBOS-TS emitter keeps parallel waves
on `Promise.allSettled` instead of queues.

### A node's constants reach its impl, and a tool only by name

Several nodes can share one `impl` and differ only in `constants`
(ops-report: three fetch nodes, one `fetch_file_content`, three
`file_id`s). The Python runtimes passed no constants at all, so every
fetch received `{}` and no implementation could tell the files apart.
Three rules now, each learned the hard way:

- **Merge in the step body, not the workflow's payload literal.** A
  `pure_function` / `external_call` step calls
  `impl(**{**payload, <constants>})` (`_py_common.impl_kwargs_expr`, used
  by dbos, python and temporal). The payload literal only reaches DAG
  callers. An agent loop calls its `loop_body` sub-nodes as tools with a
  payload the model built, so BDR's `enrich_contact_batch` lost
  `batch_size` the first time this was tried. Constants come last so the
  model cannot override a pinned value, and they never appear in the tool
  schema it sees. Judges are excluded: their input model rejects unknown
  keys.
- **TS passes them the same way, at every call site.** Cloudflare,
  Inngest and DBOS-TS call the impl directly, so there is no step body
  to merge in. Instead `payload_ts_literal` / `fan_out_ts_binding` add
  `impl_constant_entries_ts` after the inputs, and the agent-loop module
  wraps a loop_body tool call as `{ ...args, <constants> }`. Both take a
  required `with_constants` / `mcp_backed_ids`, so a new call site has to
  decide, and an MCP-backed call passes False. A TS impl reads its
  constants from `input`; hard-coding them makes the IR's value dead
  config, as on Python. `tests/test_constants_parity_ts_e2e.py` runs the
  Python parity fixture live on all three TS runtimes.
- **A constant reaches an MCP tool only when `mcp.args` names it.**
  Without `args`, the step sends its payload as the tool arguments, and a
  tool rejects arguments it does not declare (FastMCP: "Unexpected
  keyword argument"). `_common.mcp_tool_arguments` decides the arguments
  once for every runtime, Python and TS, so a tool gets identical
  arguments everywhere (`tests/test_constants_reach_every_caller.py`).
- **The impl contract is checked at emit time.** An `extracted/` function
  that takes neither the constant nor `**kwargs` raises TypeError on its
  first call, possibly after earlier steps had side effects. The Python
  adapters run `require_constant_parameters` BEFORE writing anything, on
  the module text the runtime dir will hold (`EmitWriter.landed_content`:
  the agent's source, or a user edit the writer would preserve). They
  refuse the emit naming the node, the parameter and the full path of
  the file to fix, and `rote compile` turns that into exit 1 with the
  `rote emit` retry command. `check_contracts` reports the same break as
  `constant_not_accepted`. Most committed `examples/*/runs/` predate this
  and hard-code their constants, so they no longer emit on a Python
  runtime. On Temporal the check is skipped when `extracted_module` is
  not the emitted package (an explicit `TemporalAdapterConfig`, as the
  BDR snapshot uses): those modules still receive constants and would
  fail with TypeError at run time.
- **Constants drive the logic.** An impl uses the value it is passed; a
  module-level copy is only the default for a direct call. A module that
  accepts a constant and then applies its own value turns the IR's
  constant into dead config (expense-audit did this until
  `test_editing_a_constant_in_the_ir_changes_the_rule`). Verification
  now finds it from the AST: a constant the impl takes as a parameter and
  never names in its body (no `**kwargs`, `locals()` or `vars()`) is an
  `unused_constant` review gap, repaired like a provenance gap and a
  warning if it survives (`verify.unused_constant_findings`; docs-digest
  shipped `result_limit` beside its own `RESULT_LIMIT = 5`).

Related, found by the loop test: the SDK tool runner does not serialize a
dict tool result, it iterates it, so the model saw only the keys.
`_runtime_helper._text_result` JSON-encodes a local tool's result, as the
TS twin already did.

### A fan_out element's step name must carry its index

Cloudflare and Inngest key a durable step by its **name**, so a fan_out
node emits `` `<id>[${_index}]` ``, not `"<id>"`. Reusing one name for
every element is the nastiest failure shape available here: Cloudflare
returns element 0's *cached* result for all N elements, so the run
succeeds, the output has the right length, and every entry is
identical — no error, anywhere. Inngest's memoization collapses the fan
to a single execution the same way (its e2e asserts
`count == FAN_OUT_ELEMENTS` for fan_out nodes and `== 1` for the rest,
which is what catches it). DBOS is the exception on purpose: both its
Python and TS SDKs identify a step by execution order, so repeated
calls to one registered step are already distinct — no suffix needed,
and adding one would be meaningless rather than harmful.

Second, on Cloudflare only: a **parkable** fan_out node (MCP-bound)
runs its elements *sequentially*, not in `Promise.all`, for the same
reason a parkable step leaves its parallel wave — `waitForEvent` inside
a promise combinator is undocumented and its timeout *throws*, which
would reject every other element.

### A judge's passthrough fields are copied in code, never asked for

`signature_spec.passthrough` (`{output field: input path}`) names output
fields that are input data returned unchanged (deal-monitor's scored
opportunity, a classified thread's ids). The request schema is
`passthrough_answer_schema` (those properties, their `required` entries and
any `$defs` only they used removed), and after the answer arrives
`with_passthrough` drops any passthrough key the model wrote and copies each
field from the judge's input as given (Python `model_dump(exclude_unset=True)`,
the driver `project(..., unset=False)`, TS the Zod-parsed input), so a field
the input left out is absent from the copy in every language, before the
WHOLE output schema is checked. Python runtimes' `Output.model_dump()` then
writes omitted optional fields as None, for every judge field alike. One implementation per language: the Python helper (every
Python runtime, `call_judge(passthrough=, inputs=)`, and the skill driver,
whose DECIDE shows the answer schema and merges from copies kept in the
journal) and `withPassthrough` in `_roteSchema.ts` (the TS answer schema is
computed at emit time by the Python function). The IR requires a declared
top-level target, a source walking declared object properties from a bound
`inputs:` key, identical schemas (refs inlined; annotations ignored only as
keywords, never as property names; `required`/`enum` compared as sets), and at
least one field left for the model. A source the input may omit is allowed:
the target is then absent and a required one fails the output check by
name. Before this, #264 had to make deal-monitor's prompt render the whole
opportunity so the model could retype it.

### A judge's `inputs:` must satisfy its input schema, and a fan passes one element

Every adapter builds a judge's payload from its `inputs:` keys and
nothing else (`{"thread": _item}` for a fan, plus shared params), then
validates it against `signature_spec.input_schema`. So a `required`
property `inputs:` does not bind fails *every* call. A bound key the
schema does not declare is also rejected, open schema or not: Python
silently drops it, and before the Zod emitter honored
`additionalProperties` the TS runtimes rejected every call with it. A
binding the contract cannot name is a mistake on every runtime.
Nothing caught that before run time: the IR loaded, every adapter
emitted, and the first real eval of deal-monitor failed all of its
fanned `classify_thread_step` calls on a missing `account_name`.

`Pipeline._validate_judge_input_bindings` now rejects it at load. Three
things to preserve:

- **The fanned compiler shape is the usual offender.** The compiler
  tends to flatten the element into top-level schema fields
  (`thread_id`, `messages`) while binding the list under one name
  (`threads`), or to declare a per-element value beside the element.
  Neither can work, because the grammar has no per-element reference;
  the per-element value must travel inside the element (the upstream
  step puts `account_name` on each `MatchedThread`). The rubric says so
  with the real wrong shapes; keep them there.
- **It resolves the element with `rote.ir.fan_out_element_param`**, the
  same function the emitters call, which is why that function moved out
  of `rote.adapters._common`. A validator with its own resolution rule
  could pass a pipeline the emitters dispatch differently.
- **Two exemptions, both principled.** Loop-body sub-nodes are called by
  their parent loop with model-chosen arguments, so their `inputs:`
  describe rather than bind. A legacy `signature:`-only judge has no
  schema in the IR to check. Don't widen the check to `Node.input_schema`
  on other kinds: that is observed documentation, not a contract any
  adapter enforces.

Unknown-key handling is behavioral emitted code, so it must match across
languages: Zod gets `.strict()` exactly where Pydantic gets
`extra="forbid"` (`additionalProperties: false`), strips otherwise, and a
property-less object is a free-form record on both.
`tests/test_judge_schema_parity_e2e.py` runs one schema through the
emitted Pydantic model and real `zod` and compares verdicts and parsed
values; touch either converter and run it.

`tests/test_ir.py::test_every_committed_pipeline_validates` sweeps every
`pipeline.yaml` under `examples/`, so a snapshot that would fail its
first judge call cannot be committed.

### Optional pipeline inputs are left out; required ones fail on entry

`Pipeline.optional_input_fields` (listed in `input.optional`, or a schema
property the schema's `required` omits) and `required_input_fields`
(everything else a node reads, plus the declared required) are derived,
never stored. Emitted code relies on the pair:

- **An omitted optional field is absent from the payload, not None.**
  Python used to emit `pipeline_input["f"]`, so ops-report's optional
  `report_date` raised `KeyError` whenever a caller left it out. Reading
  it with `.get` was the obvious fix and the wrong one: an explicit None
  overrides the impl's own default (`def f(d="today")`) and fails a
  judge input field that has a schema `default`. Python payloads now
  spread each input field in with the emitted `_input_field`, which adds
  the key only when the run supplied it. TS gets the same effect for free,
  since an undefined field is dropped from the JSON payload.
- **Every runtime checks the input on entry** with the emitted
  `_missing_inputs_error` / `missingInputsError`. The check names missing
  required fields, and it reports a non-object input (a run started with
  no payload at all) instead of a TypeError inside generated code. It is
  emitted whenever a node reads the pipeline input (`needs_input_check`).
  The failure must not retry. Temporal raises
  `ApplicationError(..., non_retryable=True)`, because a plain exception
  in workflow code fails the workflow task and Temporal retries that
  forever. Inngest throws `NonRetriableError`, because its retries are
  function-level.

`tests/test_pipeline_input_parity.py` runs the Python runtimes for real
both ways. `tests/test_pipeline_input_ts_e2e.py` runs the missing-input
case live on Inngest, Cloudflare and DBOS-TS, and proves Inngest fails on
the first attempt. The Inngest runner now reports the failed run's error
name and message from the dev server, where it used to show only a log
tail that held the end of a stack trace.

### An undefined fanned list must name the node, on every runtime

Emitted code routes the fanned list through a guard — `fanOutList` in
TS, `_fan_out_list` in Python — that throws naming the node id and the
IR reference. Without it the failure is `Cannot read properties of
undefined (reading 'map')` (TS) or `'NoneType' object is not iterable`
(Python): no node, no reference, inside generated code the user never
wrote. Found by running it — the first live fan_out run on Cloudflare
died exactly that way, and the cause is almost always an upstream node
that didn't return the key the IR says it does.

Both guards are emitted only when the pipeline has a fan_out node, and
they are *behavioral* emitted code like `_serialize`: if they drift, the
same broken pipeline reports differently per target. Diagnostics are
part of the emitted contract — a guard on one runtime and not another is
its own parity gap.

Related trap in the **tests**: a live e2e's mocks must return the fanned
key as a non-empty list. Three TS e2e suites had `exclusion_check_sequence`
returning `passed: []`, which made BDR's fan a correct no-op — the node
never ran and every assertion still passed. `tests/_helpers.py`'s
`fan_out_source_keys` now derives those mock values from the IR using the
adapters' own resolution, so a mock cannot disagree with the emitted code
about which input is the list.

### Eval pricing: no hardcoded prices, and two traps found empirically

`rote.eval.pricing` fetches current models + official prices at eval
time (models.dev primary, OpenRouter cross-check; 24h cache in
`$XDG_CACHE_HOME/rote/pricing.json`, else `~/.cache/rote/`, dir
overridable with `ROTE_PRICING_CACHE_DIR`; loud `PricingError` when offline — never
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
3. **Cloudflare Workflows infrastructure has no price API.** Steps and
   storage started billing on 2026-08-10. `rote.eval.infra_pricing`
   fetches the one table both Cloudflare pricing pages render, the raw
   `src/content/partials/workflows/workflows-pricing.mdx` from
   `cloudflare/cloudflare-docs@production`. It caches the raw file for
   24h in `~/.cache/rote/` and re-parses it on every load. The parser is
   strict, and both halves of that matter. A missing row or a missing
   `Workers Paid` column raises `PricingError`, because a skipped row
   prices that unit at nothing. Each cell must also match a closed
   grammar anchored at both ends. A prefix match read "per additional
   100K steps" as a per of 100, overstating the price 1000x. The
   wording has already changed once (2026-09-09). Steps per run
   (`cloudflare_steps_per_run`) are a Range: one per node, the fan width
   per fan_out node, one per HITL gate, none for `loop_body` sub-nodes.
   The fan width is the token estimate's own `fan_out_widths` (eval.yaml
   iterations, else the prior), so steps and LLM dollars never assume
   different element counts. Counting a gate's wait as a step is labeled
   a conservative assumption, since the docs don't say. Retention
   defaults to 30 days: the emitted wrangler.jsonc sets no
   `default_retention`, so the account maximum applies (limits.mdx
   footnote 2). The 2026-09-10 changelog's seven-day default for new
   Workflows contradicts that page, and this takes the larger figure.
   CPU and state size are stated priors. `rote eval --runtime
   cloudflare|rote-cloud` adds the section, as does a `compile --out` dir
   holding only `runtime/cloudflare`, or a compile that emits for
   cloudflare. Regressions in `tests/test_eval_infra_pricing.py`:
   `test_a_table_missing_any_row_is_rejected`,
   `test_an_unknown_unit_phrase_is_rejected_not_misread`.

Related invariant: the eval sidecar (`eval.yaml`, per-step agent-turn
estimates written by the compiler) is deliberately NOT part of the IR
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

### The compile verifies its deliverable, and the agent repairs it

`rote.compiler.verify.verify_deliverable` runs on every natural stop of
the agent and once more at the end: IR validation, `check_contracts`
(symbol, payload signature, Python 3.11 parse), the compile-only rules
(every `pure_function` implemented and named by a test, or a stub whose
`NotImplementedError` message says why), and `python -m pytest tests`
from the work dir. Failures go back into the SAME conversation as one
user turn, up to `MAX_VALIDATION_REPAIRS`. Things to preserve:

- **The compile-only rules stay out of `check_contracts`.** That one
  also runs at `rote emit` on hand-written and older IRs, where an
  unwritten module is the documented stub path. Moving the rules into
  it would flag every expected/ example on every emit.
- **`ast.parse(feature_version=(3, 11))` does not catch PEP 701
  f-strings** (a backslash, comment, or reused quote inside `{}`): that
  changed the tokenizer, not the grammar. `newer_syntax_problems` walks
  the 3.12+ token stream instead. Its corpus test is differential across
  the CI matrix: on 3.11 it checks the labels against the real parser.
- **Generated tests run with `PYTHONDONTWRITEBYTECODE=1`.** A repair
  can rewrite a module within the same second at the same size, and a
  stale `.pyc` then makes the fixed code test red.
- **An update verifies an overlay** of the previous output's code under
  the work dir, because an update's work dir holds only what changed.
- **A deliverable that still fails is returned, not raised.** It is the
  output of a paid run. The CLI exits 1 and skips the cloud deploy.
- **Verification never raises.** It runs inside the driver's repair loop
  (where only `DriverError` is caught) and before the work dir leaves its
  temp home, so an exception there deletes the paid work. Any crash
  becomes a `verification_error` finding; a non-UTF-8 module is a
  `not_utf8` finding.
- **A rejected IR goes to `<out>.rejected`, never into `<out>`.** A
  re-run into an existing `--out` must not replace the last good
  pipeline.yaml and the stubs the user filled in since.
- **Generated tests are untrusted code and run in an OS sandbox, or not
  at all** (`rote.compiler.sandbox`). The agent wrote them from a
  SKILL.md that may be third-party (invariant #7), and rote runs them
  automatically, up to three times. An env allowlist alone was shown to
  stop nothing: `pwd.getpwuid` names the real home, the network is open,
  and the ssh-agent socket and Keychain CLI are reachable. So:
  - macOS: `sandbox-exec` with a generated deny-by-default profile.
    Reads: the work dir; the interpreter's stdlib, its site-packages
    dirs, and as single files the binary and libpython
    (`sandbox.current_layout`, derived from `sysconfig`, never "the dir
    above the binary", which for `~/bin/python` is the home and for
    `/usr/bin/python3` all of `/usr`); `/` as a literal; `/System`,
    `/usr/lib`, `/usr/share`, dyld; a few devices. Writes: the work dir and
    a private scratch dir. Exec: literals only, the binary and a framework
    build's `Python.app` launcher. No network. A layout whose roots
    include `/`, a home, an ancestor of one, or anything holding
    `~/.ssh`, `~/.config/rote` or `~/.local/share/rote` is refused
    (`sandbox.refusal`), which means no sandbox, so unverified. `file-read-metadata` is allowed
    everywhere because path resolution needs it, so a test can learn that
    a path exists, not what it holds.
  - Linux: `bwrap --unshare-all` (network included), read-only binds of
    the system lib dirs and interpreter paths, binds of the work dir and
    scratch dir, tmpfs `/tmp`, `--clearenv`, no home. CI installs it via
    `.github/actions/bubblewrap` and lifts Ubuntu's AppArmor userns
    restriction; `tests/test_sandbox.py` fails instead of skipping when
    `CI` is set. No `unshare -n` fallback: without a mount namespace
    rebuilt the way bwrap does it, the home stays readable.
    A bind under the home is fine and normal: setup-uv's interpreter lives
    in `~/.local/share/uv/python`, a CI venv in the checkout. The property
    is that no bind exposes home *contents*: bwrap creates each bind's
    parent dirs, so inside, `~` lists an empty `.local` or `work`, and
    nothing else. The snoop test checks exactly that skeleton plus planted
    secrets in the home and the venv root; don't "tighten" it back to
    "nothing under /home is bound", which failed on every in-tree venv.
  - **The child is launched by the resolved binary, never `sys.executable`**
    (`sandbox.launch_executable`). Through a venv, Python finds its stdlib
    via `pyvenv.cfg`'s `home`, and uv writes a minor-version *symlink*
    there (`cpython-3.11-linux-...` -> `cpython-3.11.17-...`) that nothing
    binds: every sandboxed run on a uv-managed Python died with "No module
    named 'encodings'". `-S` plus the explicit PYTHONPATH means the child
    needs nothing from the venv.
  - The summary is the last line of pytest's *stdout*. A uv Python's
    ncurses prints "using dumb terminal settings." to stderr on exit (no
    terminfo inside), which used to become the summary.
  - The child runs `python -S` (no `.pth` processing) with PYTHONPATH set
    to the work dir and pytest's site-packages, and pytest's config,
    rootdir and conftest boundary pinned to the work dir: walking up
    reads the user's project, which the sandbox denies.
  - The env is still an allowlist (`verify.generated_test_env`: PATH,
    locale, an empty HOME), not a scrub list.
  - No working sandbox: the tests are skipped, status `unverified`, said
    in the compile output and in a `## rote verification` section rote
    appends to compile-report.md. `--trust-skill` runs them with the env
    allowlist only; `--no-verify-tests` never runs them.
  - **pytest is a runtime dependency of rote-cli, not a dev one.** It was
    dev-only, so every plain install skipped verification with a quiet
    `unavailable` and a real expense-audit compile exited 0 while a
    generated test failed. A missing runner is now `unverified`, loud in
    the output and the report like a missing sandbox. Regression:
    `tests/test_packaging.py::test_the_package_requires_pytest_without_any_extra`.
  - **Repair prompts restate the write-once rules** (`verify.WRITE_ONCE_RULES`):
    a real repair round made three separate one-line Edits to
    compile-report.md, each re-reading the whole context. They now say
    not to touch the report at all (rote renders it) and to end the turn
    with one sentence.
  - Not isolated: file metadata on macOS (above), CPU and memory (only the
    120 s timeout), writes inside the work dir (a test can modify the
    deliverable it tests).
  Regressions: `tests/test_sandbox.py` (each denial verified by removing
  it from the profile), and
  `tests/test_verify.py::test_generated_tests_cannot_see_credentials_or_the_users_home`.
- **One stub predicate.** `rote.adapters._common.stub_raise` decides what
  a stub is for both `rote run`'s preflight and the compile's per-node
  status (whole body after the docstring is one `raise
  NotImplementedError`). Two predicates disagreed once.
- **Outcomes cite the skill or are declared** (`rote.compiler.provenance`).
  Found by a real compile that chose block over review for a rule whose
  severity the skill never states, then wrote tests that encoded the
  choice. Every rule in it exists because a review got a guess past a
  looser version, so keep them: a citation fragment is 4+ words of prose
  (headings excluded), fragments match in order within one paragraph,
  the quote closes at the `"` ending a comment line (so it may contain
  quotes), a quote starting with `|` is a table row matched cell by cell
  (`...` any cell; 4+ words across the quoted cells, counting their column
  headers only when two or more cells are quoted, so a bare cell never passes), every outcome literal in a pure_function module needs a
  marker on its statement or in the comment directly above it or above
  the `if` that decides it (an early-return `if` ladder counts as one;
  independent checks do not, because proximity let a real compile's
  cap severity pass under an unrelated citation), labels reached by
  reference (module constants, policy tables, Enum members) are checked at
  their definition. (Open questions used to need one entry per distinct
  assumption too; rote now renders them itself, see "rote writes
  compile-report.md".) Presence guards against an accidental unmarked label, not a
  deliberate evasion: a slot whose name is not in `OUTCOME_WORDS`, or a
  label built at run time, is not seen.
  `tests/test_provenance.py` holds the gaming cases, each verified by a
  mutation.
- **What a provenance finding costs is decided by what it means**
  (the user's rule, after a real BDR compile exited 1 on 15 recorded
  assumptions). A citation not in the skill (`citation_not_in_skill`,
  `malformed_citation`, `FACTUAL_KINDS`) is an error: exit 1 after the
  repair rounds. Assumptions never fail; they are listed by node and
  `file:line` in the compile output, in compile-report.md's
  `## rote verification`, and as `provenance` in `--json` and the
  progress.jsonl summary (`Verification.provenance_report`). A missing
  declaration (`uncited_outcome`, `REVIEW_KINDS`)
  is a `review_gap`: the repair loop is sent it like an error, and if it
  survives it is a warning. Not exit 1, because a reviewer closes it from
  the listed assumptions, and failing would throw away a paid run over
  bookkeeping. `--strict-provenance` (compile and analyze, for CI) makes
  every provenance finding an error; analyze otherwise exits 0 as a dry
  run. Quote marks are normalized (`'`, `"` and curly quotes are one):
  the BDR compile's one "fabricated" citation was verbatim except that
  it wrote the skill's `"BDR do not contact"` in single quotes inside its
  own double-quoted citation.

### The claude driver repairs by resuming, and a resumed result is cumulative

`ClaudeDriver` sends a repair as `claude -p <instruction> --resume
<session_id>` with the same flags. Measured on claude 2.1.302: a resumed
run's `total_cost_usd` and `modelUsage` cover the WHOLE session (the
first run's plus its own), while `num_turns` and `duration_ms` are per
invocation. That is observed, not documented with a version floor, and
least likely to hold after an abnormal exit, so `_session_metadata`
checks it: non-decreasing costs mean cumulative (use the last result),
anything else is summed per round (`usage_basis` says which), and a
round with no result marks `usage_complete` False. The usage gotcha
above applies: a total that silently drops a round reads as a small
number.
`--resume` looks the session up by cwd, so every invocation of one
compile runs with `cwd=work_dir`.

### The compiler's `claude -p` child is isolated from the user's setup

`rote.inference.CLI_CONTEXT_ISOLATION_FLAGS` (`--setting-sources ""`,
`--strict-mcp-config`, `--settings '{"autoMemoryEnabled": false}'`), then
`--tools <allowlist>`, and `cwd=work_dir`. The context flags are the SAME
tuple the emitted judge and agent-loop lanes use (they add `--tools ""`
instead); the driver imports it from the helper, like
`build_subscription_env`, so there is one copy. Each one was found by a real compile: a user's Vale
prose hook (em dashes in `compile-report.md`) cost 12 of 48 turns in
one-line edits. The same user settings granted Bash despite the
allowlist. And run from this repo, the agent globbed into
`examples/<skill>/runs/` and read an earlier compile of the same skill as
a template, so the output depended on where the command was typed.
`--tools` takes a VALUE here, as in the judge helper. When measuring the
compiler, run it against a copy of the skill outside the repo anyway:
an older installed rote has none of this.

### A skill run is gated by permissions, not by an allowlist

`rote run <skill>`, `rote baseline`, `compile --baseline`, `rote eval
--run` and the bench's fidelity references all build their `claude -p`
child through `rote.eval.empirical.skill_run_isolation_args`. A
walkthrough found the old shape leaking both ways: `--strict-mcp-config`
was only added when rote had servers, so an empty registry showed the run
every server the user has in Claude Code (work servers and claude.ai
connectors included), and `--allowedTools` only pre-approves, so under the
user's `defaultMode: auto` the run called a server the gate excluded and
used WebSearch while the summary said "read-only MCP gate". Verified on
Claude Code 2.1.308, each with a real run:

- **Always strict**, with an empty `{"mcpServers": {}}` config when nothing
  is wired (init event: `mcp_servers: []`). Plus the shared
  `CLI_CONTEXT_ISOLATION_FLAGS`, so the user's settings and permission
  mode never load.
- **`--permission-mode dontAsk`** denies anything `--allowedTools` does not
  name. The result envelope lists each refusal in `permission_denials`,
  and the run reports them. **`--tools Read,Write,Edit,Glob,Grep`** removes every other
  built-in (Bash, WebSearch, WebFetch) from the model's context.
- **File access is scoped** (`skill_run_file_rules`): `Read(./**)`,
  `Read(//<abs skill dir>/**)` and `Edit(./**)`, nothing bare. A bare
  `Read,Glob,Grep` let a probe read a directory outside the run, and a
  third-party SKILL.md could pass `~/.ssh` out as a read-only tool's
  argument. Glob and Grep are checked against Read rules (a `Glob(path)` or
  `Write(path)` rule is accepted and never matches, and `Edit(path)` covers
  Write). `//` starts an absolute path, `./` is the cwd.
- **`--disallowedTools`** names the wired servers' tools the read-only gate
  excludes (explicit ids, never `mcp__<server>__*`: deny beats allow). A
  registered server that contributes no callable tool is not wired at all.
- **A run that never called a server its skill names is degraded**
  (`missing_mcp_servers`, from `tool_grounding.servers_named`): `rote run`
  exits 1 unless `--allow-missing-servers`, and every summary names the
  server and its fix. Only explicit forms count (`` `x` MCP server ``,
  `mcp__x__tool`, a bundled `mcp-tools.json`, and for a registered name
  "X MCP" or a tool-table mapping), because a false entry fails a correct
  run: a tool table alone reads expense-audit's caps table as a
  `category_caps` server. The empty-registry docs-digest run that "worked"
  wrote its digest from memory.

### A failed `claude -p` compile is still a paid one, and the output cap resumes

A real invoice-push compile died after 28 minutes and 8 turns with
`Claude's response exceeded the 32000 output token maximum`, and its cost
appeared nowhere: the driver raised without reading the result envelope,
which every such failure still carries. Three things now hold:

- **`DriverError.metadata`** carries the failed session's usage (same
  keys as `DriverResult.metadata`, via `_session_metadata`). The
  orchestrator prints it in the error and emits a `usage before the
  failure:` log line, so progress.jsonl has it too.
- **`CLAUDE_CODE_MAX_OUTPUT_TOKENS=64000`** in the compiler child's env
  (`ClaudeDriver._build_child_env`), `setdefault` so a caller's value
  wins. Unset, Claude Code uses 32000 for a model it cannot resolve;
  64000 is the default compile model's own maximum, and the CLI lowers a
  value above a model's cap to the cap. Raising it also shrinks the
  context left before auto-compaction, so don't push it to 128000
  without measuring.
- **The cap is resumable, once per compile.** The session is intact, so
  the driver resumes it with `OUTPUT_CAP_RESUME` (write large files in
  pieces) and then takes the normal file-first path. Once, not per
  round: a model that hits the cap twice is not going to stop.

---

## Canonical examples to imitate

When adding new code, the fastest way to stay consistent is to
copy the shape of an existing canonical example.

| What you want to do | Imitate |
| --- | --- |
| Add a new node kind to the IR | `rote.ir.Node` — add an enum variant, a kind-specific required field, and a branch in `_validate_kind_specific_fields` |
| Add a Python-emitting runtime adapter | `src/rote/adapters/temporal.py` (~570 lines) |
| Add a TS-emitting runtime adapter | `src/rote/adapters/cloudflare.py` (~750 lines) — includes the JSON-Schema-to-Zod converter and signal-name validator |
| Add a subprocess-based driver | `src/rote/compiler/drivers/claude.py` + `tests/test_claude_driver.py` |
| Add an in-process driver | `src/rote/compiler/drivers/anthropic_api.py` + `tests/test_anthropic_driver.py` |
| Add a new CLI subcommand | `_cmd_emit` in `src/rote/cli.py` (simple) or `_cmd_compile` (which calls into the Compiler orchestrator) |
| Test a Python-emitting adapter end-to-end | `tests/test_temporal_e2e.py` (uses Temporal's `WorkflowEnvironment.start_time_skipping`) |
| Test a TS-emitting adapter end-to-end | `tests/test_cloudflare_e2e.py` (real `npm install` + `tsc --noEmit`; `@pytest.mark.slow`) |
| Test the CLI via subprocess | `tests/test_cli.py::test_subprocess_emit_bdr` |
| Add a rubric file | Any file in `skills/rote-compile/references/` — keep them 200-300 lines, concrete, with BDR examples for every point |

**Never tune against `examples/holdout/`.** Those skills and their gold
pipelines are the held-out half of `rote bench` (see
[`examples/holdout/README.md`](examples/holdout/README.md)). Rubric or
prompt work is judged on the five top-level examples, then checked
once against the holdout with `--examples examples/holdout`. Don't
cite holdout skills as rubric examples either: that leaks the answer
key into the compiler.

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
3. **Emitted code touches MCP only where the IR declares it.** There
   are exactly two declarations, and no third: a node's `mcp:` binding,
   and an `agent_loop`'s `tools:` (which the IR documents as MCP tool
   names — a loop that declares any pulls in the MCP helper even when
   no node in the pipeline carries a binding). Everything else never
   references MCP in any adapter or language, enforced by AST/text
   tests that **name** the legitimate sites rather than skipping files
   that happen to mention MCP — so an unexpected reference still fails,
   and an expected site that stops referencing MCP fails too
   (`tests/_helpers.py::assert_no_mcp_in_ts`'s `expected=`). Both
   declarations emit a working, *never-interactive* client call — auth
   problems raise/park, they never open a browser from workflow code.
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

## Tests vs. evals: the dividing line is inference

**Anything that does not require inference is a test.** Automated, run
on every push, and kept fast.

**Anything that costs tokens is an eval.** Run deliberately and
periodically — `rote eval --run`, the compiler against a real skill,
a live gateway judge. Never wired into CI.

This is not a style preference. A token-spending check in an automated
suite bills someone on every push, gets slower as it grows, and fails
for reasons that have nothing to do with the change under test. Both
halves of the split are enforced, not just documented:

- `tests/conftest.py::pytest_configure` changes `os.environ` once, before
  collection (`tests/_inference_guard.py`): every vendor credential
  scrubbed, every `rote.isolation.STATE_PATHS` entry and the four XDG
  base dirs moved under a session temp dir (so a deleted override or a
  child process that drops `ROTE_*` still misses the real files; uv's
  Python and cache dirs are pinned first so they stay shared), and each
  PATH entry holding `claude` / `codex` replaced by a read-only mirror of
  that dir without them (so `node`, `uv` or `rote` beside them still
  resolve). HOME is not moved: npm and docker keep their state there,
  and no rote path reads the home except through an XDG dir. The PATH half is the one that
  matters: `claude -p` authenticates from an OAuth session, so having
  no API key is *not* protection. Not finding the binary is. Two DBOS
  e2e suites burned real subscription inference on every run until it
  was caught by wall-clock timing. It used to be a per-test
  monkeypatch of `shutil.which`, and that leaked twice: a
  `monkeypatch.undo()` removed it, and a child process never saw it, so
  a test's `python -m rote compile` subprocess found `claude`, billed
  $0.29 and wrote into the real app registry. Never move it back into a
  per-test fixture, and never use a failing `claude` stub instead of the
  mirror: a stub makes `which("claude")` succeed, which rote reads as
  "available".
- `conftest.pytest_runtest_call` fails (never skips) any test during
  which a bare `claude` / `codex` resolves, in-process or from a child
  shell, before and after its body.
- `tests/test_no_inference_in_tests.py` verifies the guard itself,
  because its failure mode is silent spending that no other assertion
  would notice.
- Only *bare-name* lookups are blocked, so a test that writes a fake
  `claude` stub and passes its absolute path still works — a guard that
  forced tests to route around it would soon be disabled.

Consequences worth keeping in mind:

- **An e2e test is still a test.** The TypeScript suites run real
  `npm install`, real `wrangler dev`, a real Inngest dev server and a
  real Postgres — no inference, so they are automated (see the
  `TypeScript e2e` / `DBOS-TS e2e` CI jobs). `@pytest.mark.slow` says
  what *toolchain* a suite needs, never whether it should run.
- **Every npm install in a test goes through `tests/_node_trees.py`.**
  It installs once per distinct dependency set (the dependency fields
  of `package.json`, install scripts, extra npm args, node and npm
  versions) and symlinks each app's `node_modules` to that tree. Per-app
  installs used to take 8.7G per CI run, which overflowed ubuntu-26.04's
  quota-limited `/tmp`. A slow test's `rote run` installs go the same
  way (conftest swaps the runners' `ensure_npm_install`). A raw
  `npm install` in a test fails `tests/test_node_trees.py`, because it
  would write into a tree other tests share.
- **Emitted judges and agent loops get stubbed, not called.** The live
  suites point `ROTE_BASE_URL_<NODE_ID>` at a local Messages stub and
  set a fake key. That is what makes it possible to prove the agent-loop
  machinery end to end without paying per run.
- **Speed is a feature of the test half only.** The fast suite is ~18s;
  keep it there. Evals are allowed to be slow and expensive because
  nothing blocks on them.

---

## Testing discipline: make assertions capable of failing

Five real defects in this repo shipped past a green suite because a
test observed something *adjacent* to the behavior. This is the single
most common failure mode here, so it gets its own section.

| Defect | What the test checked | What it should have checked |
| --- | --- | --- |
| bare `--tools` killed the whole subscription lane | `"--tools" in command` | the flag's **value** |
| DBOS e2e mocked agent loops via an unreachable branch | that the overlay file was written | that the mock was actually *called* |
| BDR's fan was a no-op in 4 live e2e suites | that the workflow completed | that the fanned node ran **N** times |
| per-node `timeout:` ignored (temporal, cloudflare gates) | that *a* timeout was emitted | that the **declared** one was, and the default was not |
| `1h` emitted as `"1 hours"` | nothing — no per-node timeout test existed | — |

Three habits that catch these:

- **Assert the value, not the presence.** A flag, key, or node id
  appearing in the output says nothing about what it was set to.
- **Give a negative half.** Pair "the declared value is present" with
  "the default is absent" — otherwise emitting both still passes.
- **Make fixtures non-degenerate.** An empty list makes a fan, a loop,
  or a retry a *correct* no-op that no assertion can catch. Fixture
  collections that drive iteration count carry >1 element
  (`tests/_helpers.py::FAN_OUT_ELEMENTS`), and their values are derived
  from the IR (`fan_out_source_keys`) so a mock cannot disagree with the
  emitted code about which input is which.

**Verify a new test by breaking the code.** Not by inspection — apply
the regression the test is meant to catch and confirm it fails. Any
test written for a bug fix should be run against the *unfixed* code
once. When the fix spans runtimes, check the split: the fan_out parity
suite fails on the five broken runtimes and passes on `dbos`, which
already worked, proving it recognizes correct behavior rather than just
matching new strings.

A periodic mutation sweep is the systematic version, and it works well
here because emission is pure template substitution — mutate a module,
run `pytest tests/`, and any mutation that survives is an untested
behavior. Three sweeps so far, 64 mutations, 48 caught on the first pass:

- **Adapters** (18): retry policy, timeouts, step naming, payload
  threading, MCP allowlist narrowing, the `ANTHROPIC_API_KEY` scrub,
  invariants #4 and #7. Gaps found: per-node `timeout:` was ignorable on
  both temporal and cloudflare gates with nothing failing.
- **Drivers / inference / MCP / probe** (17): the flags that keep
  `claude -p` cheap, `is_error` inside a zero exit, fastmcp's
  `result.data`, probe's never-guess rule, and the cost-critical driver
  defaults.
- **`rote.eval` / `rote.config` / `rote.cli`** (29): dollar arithmetic,
  tier detection, the read-only MCP gate, the four-layer config
  precedence, and the login-aware compile default. Config precedence and
  pricing tier detection were already airtight — every mutation died.
  Six gaps found, all in the two places where a wrong number or a wrong
  exit code looks like success (see below).

Two lessons from the second sweep worth keeping:

- **The agent-loop CLI path is the weak twin.** `_runtime_helper` carries
  two copies of the `claude -p` flag block (judge, agent loop), and
  mutations died on the judge copy while surviving on the loop copy —
  `--setting-sources` and `is_error` were covered on one and not the
  other, and `--system-prompt` on neither. That asymmetry is exactly how
  the bare `--tools` bug shipped. When you touch one copy, test both.
- **A symbolic assertion cannot pin a constant.**
  `args[args.index("--model") + 1] == DEFAULT_MODEL` verifies the flag is
  wired up and says nothing about the constant, because both sides move
  together. `DEFAULT_MODEL` could be flipped to Opus (~5x cost) and
  `DEFAULT_MAX_TURNS` halved (the original `error_max_turns` failure)
  with the suite fully green. `tests/test_cost_defaults.py` pins the
  *property* — not-Opus, `>= 60` — so a legitimate model bump passes and
  the expensive regression fails.

Three more from the third sweep:

- **A self-comparison is the same tautology in disguise.**
  `pipeline_cost_usd` had `cost.high == pytest.approx(expected_high)`
  recomputed from the fixture rates and killed every mutation.
  `agent_run_cost_usd` was only ever compared against *itself*
  (`with_cache.mid < without_cache.mid`) — an ordering that holds under
  any transformation applied to both sides, so the divisor could become
  1e3 instead of 1e6, or the cache-write premium could vanish, with the
  suite green. Cost numbers are what a user makes a spending decision
  on; assert the dollars.
- **Asserting the loop bound leaves the per-call figures unpinned.** The
  `agent_loop` estimate asserted `calls.high` and nothing else, so both
  the token and the wall-time lines could drop
  `agent_loop_turns_per_iteration` (a 3x understatement) undetected —
  in the turn-dominated regime the loop-aware model exists to capture.
- **Strictness is a property of a layer, not of one command.** Four
  commands call `load_layers()`; only `rote config` asserted exit 2, so
  compile's handler could `return 0` — reporting success while doing
  nothing. The replacement test parametrizes over all four, and the
  next command to read config fails there instead of shipping a silent
  fallback.

Watch for **equivalent mutants** — the fan_out tier ordering in
`fan_out_element_param`, and `args.no_deploy` in `_cmd_compile`'s
cloud-vs-local branch, both survive because they are provably
unobservable, not because they are untested. Both are commented in
place; confirm the claim empirically before writing one off, since
"probably equivalent" is exactly what a real gap looks like.

---

### The suite never writes into examples/

The committed runtime snapshots under
`examples/bdr-outreach/expected/runtimes/<runtime>/` are compared, not
regenerated: adapter suites emit into `tmp_path`, and
`tests/test_committed_snapshots.py` checks that a fresh emission (with
the same `extracted/` sources `rote emit` reads) reproduces each one
byte for byte. The suites used to emit straight into those dirs, so every
pytest run rewrote committed files, and without the example's sources it
replaced the real implementations with stubs. That dirtied the
merge-queue worktree and blocked merges. A session hook in
`tests/conftest.py` now fails the run if anything under `examples/`
changed (`__pycache__` and `node_modules` excepted). When an adapter
change is intended, regenerate with the `scripts/isolated-rote emit`
command the failing snapshot test prints, and commit the result.

### `rote skill run` runs only what the user installed, in an allowlisted environment

A Claude Code allow rule pre-approves `rote skill run <name> *`, so the
command must approve exactly the files the user vouched for
(`rote.skill_target.trust`, `launch`, `commands`; design in
`docs/design/compiled-skill-target.md` section 2). Things to keep:

- **Every file is hashed, and anything else is refused.** `rote skill
  install` records the hash of every file in the dir (after checking it
  matches `.rote-manifest.json`), in the app registry entry's `trust`. A
  run refuses a changed, new, missing, symlinked or special file, and
  `__pycache__` counts as new: an unchecked `.pyc` with the source's mtime
  shadows a listed `.py`. The driver runs WITHOUT `-B`: its code comes over
  a pipe (below) and is never cached, so no `__pycache__` reaches the skill
  dir, while `-B` made every block recompile fastmcp on a `uv tool install`
  (no bytecode shipped, 1.0 s against 0.5 s cached).
  `test_the_driver_keeps_rotes_bytecode_and_writes_none_into_the_skill`
  pins both halves.
- **Every block pays the start-up, so keep `skill run` light.** The console
  script is `rote.entry:main`, which hands `skill run <name> ...` to
  `launch.run` without importing `rote.cli` (whose parser imports the
  compiler, the adapters and the IR). Keep `rote.skill_target`'s package
  import free of the IR, and keep vendor SDKs out of `_build_parser`
  (`anthropic` cost 0.5 s on every command). The child leaves with
  `os._exit` after flushing, skipping an 85 ms teardown. Measured on
  docs-digest (2026-10-24): what is left is fastmcp's own import (~0.5 s)
  and the MCP round trips. Each request opens a new TCP+TLS connection
  (the SDK closes an SSE response before its last chunk, so HTTP/1.1
  cannot reuse it), and each call also pays a `server/discover` probe and
  a session `DELETE`.
- **The env is an allowlist, never a scrub list** (`launch.child_env`).
  Endpoint overrides, `ROTE_MCP_READ_ONLY*`, `ROTE_SKILL_COMMAND` (the
  next command every block prints), base URLs, `XDG_*`, proxies and CA
  bundles are dropped. Header-template variables pass only when the bound
  registry names them for this plan's servers.
- **State paths are bound to the install, not merely read.** The MCP
  registry, login store and run dir each widen trust when forged (a server
  declared read-only, someone else's login, a gate nobody answered).
  Their overrides stay, for tests and `isolated-rote`. A run whose paths
  resolve elsewhere than at install is refused. `ROTE_APPS_PATH` is the
  root and is not bound: whoever sets rote's env can set `PYTHONPATH` too.
- **The running code never touches disk.** `trust.scan` keeps each file's
  bytes. `launch.frame` sends them, plan.json included, over an inherited
  pipe to `launch.BOOTSTRAP` (a `-c` constant, never a file), whose
  meta-path finder serves the driver and `extracted/` from those bytes with
  the skill dir's paths as `__file__`. The skill dir's `sys.path` entry is
  disabled. Two reviews broke the earlier shapes: swapping an extracted
  module in the skill dir after the first `rote: calling` line, and then,
  since the agent is the same uid, rewriting rote's private 0700 snapshot
  (`test_no_file_of_the_running_code_exists_for_the_agent_to_swap`). Only a
  CA bundle still lands on disk, because TLS reads a file.
- **Write consent is a driver parameter, from the registry only.** The
  bootstrap (`launch.BOOTSTRAP`) passes `allow_writes` to `main`. Never add
  a flag or variable for it: `uv run scripts/run.py` would take it too.
  The consent is bound to the plan's hash, and any rebind drops it.
- **Everything that widens trust needs a person, not `--yes`**
  (`commands._require_person`, `PersonRequired`, exit 2): every `trust`
  that records anything, an install that changes an existing record or
  replaces another dir's, and any install relying on a server registered
  read-only. That last one counts because a read-only declaration lets every
  tool on the server run, so the set of read-only servers (name and URL) is
  bound in the record like the state paths. A detected caller cannot be the
  person even with `--agent-mode no`. It is a speed bump, not a barrier: an
  agent with a pty and its variables unset looks like a person.
- **Proxies and CA bundles are bound, not dropped.** The proxy variables are
  recorded at install and passed as recorded. A CA bundle is recorded by path
  and hash and handed to the driver as rote's private copy. A proxy with a
  non-loopback `http://` endpoint is refused.
- **A kept copy is checked against its own name.** `trust._copy` returns
  nothing when the bytes no longer hash to the digest. Then the summary says
  so and the whole current file is shown, because a diff against a planted
  copy would show the person a harmless change.
- **The allow rule is its own Action, never `--yes`**
  (`--save-allow-rule`), written only to `<project>/.claude/settings.local.json`,
  the one file measured to work. The project defaults to the git root
  (Claude Code 2.1.211+ keeps the file there). Claude Code keeps the file out
  of git only when it writes it itself, so the Action warns when `git
  check-ignore` says it would be committed. Tests always pass
  `--claude-project <tmp>`, because the default target is the cwd's repo,
  which under pytest is this one.
- **`record_app` keeps every entry's `trust`.** It used to rebuild entries
  from four fields, which would have dropped every install record on any
  emit.

Regressions: `tests/test_skill_commands.py`, every assertion verified by a
mutation.

### Repo maintenance runs rote through `scripts/isolated-rote`

A plain `rote emit` appends its output dir to the developer's real
`~/.local/share/rote/apps.json`, and `rote mcp login` later walks that
list to release parked runs. Agents regenerating snapshots kept adding
`examples/` and temp dirs to it. `scripts/isolated-rote <args>` runs the
repo's rote with every per-user state path moved into a temp dir that is
deleted afterwards (`ROTE_ISOLATED_STATE=<dir>` keeps it). The list of
paths is `rote.isolation.STATE_PATHS`, the same one `tests/conftest.py`
and the bench harness use. `tests/test_isolated_rote.py` derives the set
of `ROTE_*_PATH`/`_DIR`/`_CONFIG` overrides from `src/` and fails when a
new one is missing from the list. The wrapper hides your MCP logins too,
so use plain `rote` for a run that needs a real MCP server.

### Onboarding commands: every side effect is a confirmed Action

`rote init` (and `rote setup`, `rote mcp import`, `rote try` as they land)
run on `rote.interact`; the architecture is
[`docs/design/first-run.md`](docs/design/first-run.md). Three rules:

- **Decide interactivity once, before any prompt.** `Mode.resolve` at
  entry, then `prompter_for(mode)`. Never call `input()`, `isatty()` or
  questionary in a command. prompt_toolkit reads a piped stdin as
  keystrokes and returns an answer with no error, so a prompt reached
  without a terminal is a silent wrong answer
  (`tests/test_init.py::test_piped_answers_never_reach_a_prompt`).
- **Every question names its flag**; the command checks the flag first.
  Nobody at the terminal: `--yes` takes defaults, anything else raises
  `PromptRequired` naming the flag (exit 2, `action_required` JSON).
- **Every write, contact, launch or spend is an `Action` in a `Plan`**,
  previewed and confirmed (`confirm_plan`) before `apply_plan` runs it.
  Replacing a file needs `--force` beyond `--yes`, and spending needs the
  `Spend`'s own flag: `--yes` never spends money. A `Contact` holds a
  bare host, never a URL. A paid `Launch` must sit beside its `Spend`.
  Writes go through `write_file_action(read=snapshot(path), ...)`, which
  refuses at apply when the file no longer matches the caller's own read
  (a confirmed "create" must not replace a file that appeared while the
  person read it, and a merge must be checked against the bytes it
  merged into, not a second read).

`rote mcp import` (`rote.setup.mcp_import`, design in first-run.md) keeps
three rules:

- **Never preselect what needs a yes.** Only `--servers-for` preselects,
  and only `preselectable` candidates off the user's `work_domains`. Every
  other reason a candidate is not preselectable, except being inactive in
  the client, needs a per-server opt-in (`opt_ins`: `--allow-url-secret`,
  `--allow-untrusted`, which also covers any reason this version does not
  name), so a client source that adds a new not-preselectable state is
  safe with no change to import. A bare `--yes` declines all of them.
- **Additive, and collisions never write over.** A taken name gets the
  first free `<name>_N`; `--rename` picks one, `--replace` (plus `--force`
  without a person) overwrites. An endpoint registered under any name is
  "already registered".
- **`work_domains` is a list setting: every layer adds, none removes**
  (`resolve_list`), so a project `rote.yaml` cannot switch off the user's
  own. Only the user answers the question (their config holding the key,
  `--work-domain`, or the prompt); project and env values add domains but
  never answer it and are never saved. It is asked once, with no default (a
  default cannot know which domains are work), only when an answer matters,
  and saved by editing just its line (`set_in_place`).

A registry header's `${NAME}` is filled at connection time by one rule in
every helper (`expand_headers`, TS `_HEADER_TEMPLATE_TS` in the Node
helper); unset or empty is a `RoteMcpConfigError`, never an empty header.
No filled value goes on a command line: the agent loop's `--mcp-config`
(argv) carries a `headersHelper` instead: `python -I -c <the MCP helper's
source, from its module loader>`. Never exec `_rote_mcp.__file__`: under
`rote skill run` the module came from checked bytes, and the file at that
path is the unchecked one an agent can rewrite.
`tests/test_mcp_header_template_parity_e2e.py` pins the twins together.

Client config readers (`rote.clients`) only read: never run the client,
contact a server, or resolve a placeholder, and display hosts, never URLs
or header values. A credential-shaped URL path sets `risk`, which keeps
the server out of every default until the user opts in by name. They reach the machine only through `ClientContext`;
`ROTE_CLIENT_HOME_DIR` (in `STATE_PATHS`) keeps tests and
`scripts/isolated-rote` away from the user's real AI-client configs.
One module per client (12 today) shares `rote.clients.common`; a new client
is a module plus a line in `SOURCES`. Three traps. A variable a source reads
to find files must join `CLIENT_DIR_VARS`, or isolation leads back to the
real files (`tests/test_clients_registry.py` derives the set and fails).
YAML goes through `common.Reader`'s YAML 1.2 loader, because PyYAML reads a
Goose extension named `off` as `False`. A client whose entry wins by name
even when disabled marks it with `switch_off` after `rank`, never before,
or a disabled project entry uncovers the user's. A `project` candidate
starts at trust `unknown` (fail closed). A source sets `trust_verified` only
after reading a gate that passed, and otherwise `trust_unknown` or
`trust_failed` with its own note. The registry test fails on a project
candidate left at the default. `preselectable` and `needs_opt_in` are what
`rote mcp import` reads.

`rote.skills.discover` (skill discovery and ranking) follows the same
rules: files only, through `ClientContext`, enabled plugins from
`claude_code_layout` (one resolution for MCP servers and skills), and no
skill text reaches the terminal unsanitized (`_display`). Keep it cheap:
it reads hundreds of skills, so a check it runs per skill must not grow
into a full parse (`system1.parse.tool_registry` exists for that).
Skill text is untrusted and unbounded: read it through `rote.skill_files`
(size caps checked by `stat` first, references kept inside the resolved
skill dir, no symlinked-directory walks), never with a bare `read_text` or
`rglob` over `references/`. Compile's tool grounding and the System-1
parser already do; a new reader of a whole skill should too.

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

### Running the real compiler

```sh
scripts/isolated-rote compile examples/bdr-outreach/skill \
  --runtime temporal \
  --out /tmp/bdr-compiled
```

~13 minutes wall clock on BDR. Expect 30-40 turns with Sonnet 4.6.
The output landing at `/tmp/bdr-compiled` (or wherever you point
`--out`) is what you'd commit as a new regression snapshot if the
rubric or IR changed in a way that requires re-compiling.

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
- [ ] If the rubric or IR changed materially: re-run the compiler
      on BDR and diff the snapshot
- [ ] Commit message explains the *why*, not just the *what*

---

## What's stubbed vs. working

Don't waste time debugging stubs. These are intentional.

**Intentionally stubbed:**

- The BDR example's `extracted/*.py` modules raise
  `NotImplementedError` — users fill them in with real API client
  code. The compiler leaves vendor calls (`external_call` impls) as
  contract-documented stubs unless an `mcp:` binding or probe ground
  truth makes them real. `pure_function` nodes are NOT in this column:
  the compiler implements and tests them (see the verification gotcha).

**Working end-to-end:**

- `fan_out` on ALL SIX runtimes: a node marked `fan_out: true` is
  invoked once per element of its bound list, and the result binds as a
  list in input order. Which input is the list is resolved by
  `rote.ir.fan_out_element_param` (fan_out edge marker >
  incoming-edge source > only node-bound param > with no node-bound
  param at all, the only pipeline-input field that can hold a list;
  ambiguity is an emit-time error, never a guess). It lives in the IR module on
  purpose: two adapters disagreeing about which input is the list would
  make one pipeline mean two different things, and the judge-binding
  validator must agree with them too. Per
  runtime: DBOS enqueues one durable step per element; python uses
  `pool.map`; temporal `asyncio.gather` over one activity execution
  each; Cloudflare and Inngest `Promise.all` over per-element steps;
  DBOS-TS `Promise.allSettled` + `unwrap`. See the fan_out gotchas
  below for the two traps.

- `agent_loop` on ALL runtimes. Python got the real loop first; the
  three TypeScript runtimes now emit one too — MCP tools bound through
  whichever MCP helper that runtime already emits, `loop_body` sub-nodes
  bound as callables, and the whole thing handed to `runAgentLoop` in
  the verbatim `signatures/_roteInference.ts`
  (`rote.adapters._ts_common.ROTE_INFERENCE_HELPER_TS`). Proven live in
  `tests/test_ts_agent_loop_e2e.py`: real fastmcp server, real emitted
  module compiled by the emitted toolchain, real Anthropic SDK tool
  runner against a local Messages endpoint.
- IR load + validate
- `rote emit` (IR → runtime code; DBOS default), with hash-guarded
  re-emission: every adapter writes through
  `rote.adapters._common.EmitWriter`, which records what rote wrote in
  `.rote-manifest.json` and, on re-emit, preserves any file the user
  edited (fresh content lands in a `<name>.new` sibling and the CLI
  reports it). Never write emitted files with bare `write_text` in an
  adapter — route through the writer. Emission also sources the agent's
  REAL `extracted/<module>.py` implementations found beside the
  pipeline.yaml verbatim into the runtime dir (dbos, python and
  temporal adapters, `rote.adapters._py_common.resolve_extracted_source`);
  IR-derived stubs are only the fallback. Temporal writes its package
  through the shared `write_extracted_package` (python uses it too) and
  its activities call plain functions in a worker thread and await
  coroutine functions, since agent-written modules are plain. Only the
  BDR snapshot points temporal at a user-maintained package, through an
  explicit `TemporalAdapterConfig(extracted_module="expected.extracted")`.
  Known gap: TS runtimes can't consume Python modules at all (the
  compiler would have to emit TS implementations).
- `rote compile` (SKILL.md → IR → Temporal code), stamping a
  `provenance.json` sidecar (section hashes; see `rote.skill_source`).
  Every run streams NDJSON progress to `<out>/progress.jsonl`
  (`--progress-file` just moves it) and prints the `tail -f` watch
  command as its first stderr line — agents driving a run relay that
  line to the human (AGENTS.md).
- `rote compile --update` — incremental re-compilation: diffs the
  skill against the stamped provenance, re-derives only nodes whose
  source sections changed, enforces that unchanged nodes' ids survive,
  and merges file-level so user-filled stubs are kept. No section
  changes → no agent run at all.
- Per-node inference overrides in emitted judges: `ROTE_MODEL_<ID>` /
  `ROTE_BASE_URL_<ID>` (plus `ROTE_TEMPERATURE_<ID>` when the IR sets
  one) env vars beat the IR defaults on all four runtimes, and
  `signature_spec.base_url` reaches any OpenAI-compatible endpoint.
  Node.source (provenance) is excluded from the pipeline hash on
  purpose — metadata must not re-version in-flight workflows.
  **Proven config — judges through a Cloudflare AI Gateway** (no
  Anthropic key at all; Unified Billing pays provider rates from CF
  credits and logs every call's tokens + cost):
  `ROTE_BASE_URL_<ID>=https://api.cloudflare.com/client/v4/accounts/<ACCT>/ai`,
  `ROTE_MODEL_<ID>=anthropic/claude-sonnet-5`, and the CF API token in
  `ANTHROPIC_AUTH_TOKEN` — the SDK sends it as `Authorization: Bearer`,
  which is what that endpoint wants; passing it as `ANTHROPIC_API_KEY`
  (→ `x-api-key`) gets a 401, contradicting Cloudflare's own JS
  example. Two more traps: only *current-generation* model ids resolve
  (`anthropic/claude-sonnet-4-5` from the CF docs 404s), and `@cf/*`
  Workers AI models are rejected by the `/v1/messages` route entirely —
  they only work on `/v1/chat/completions`. Routing to a *named*
  gateway needs a `cf-aig-gateway-id` header, which emitted clients
  can't inject; the account's default gateway needs no header.
- All three compiler drivers: `ClaudeDriver` (`claude -p`),
  `CodexDriver` (`codex exec` — workspace-write sandbox, no env
  scrubbing since Codex login is only overridden by `CODEX_API_KEY`,
  smoke-tested against the real CLI), `AnthropicApiDriver`
- `rote analyze` — the `plan` to `compile`'s `apply`: runs the
  compiler, then prints a structural report (node-kind breakdown,
  roteness matching `rote eval`, HITL gates, targetable runtimes)
  instead of emitting runtime code. `--json` for machines; `--out`
  keeps the IR for a later `rote emit`.
- MCP requirements manifest: `Pipeline.required_mcp_servers` (derived,
  never stored — same rationale as `requires_durable_execution`) feeds
  an `mcp_servers` entry — `{server, nodes, tools, auth}` — in
  `analyze`/`emit`/`compile` output (human, `--json`, and the
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
  emit/compile time). Release channels differ per runtime: DBOS =
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
  Detection understands both the `compile --out` layout
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
- Layered defaults (`rote.config`) + `rote init` + `rote config`:
  `runtime`/`deploy`/`agent`/`model` resolve flag > `ROTE_*` env >
  project `rote.yaml` (walk-up discovery, stops at `.git`;
  `ROTE_PROJECT_CONFIG_PATH` overrides) > user
  `~/.config/rote/config.yaml` (`ROTE_CONFIG_PATH` overrides —
  conftest isolates both suite-wide) > the command's built-in. Config
  files are strict: unknown key or invalid value = loud exit 2, never
  a silent fallback. `rote init` is the interactive wizard that writes
  them (the ONLY interactive command besides login; exits 2 without a
  TTY; I/O injected for tests — and its `input_fn` default resolves at
  call time, not def time, or monkeypatched `builtins.input` never
  lands). `rote config` prints each key's value + winning source.
  Compile deploy semantics: config `deploy: rote-cloud` + logged out
  fails fast BEFORE the agent spends money; an explicit `--runtime`
  flag downgrades a configured cloud deploy with a warning instead of
  blocking; contradictory config (non-cloudflare runtime + rote-cloud
  deploy, both from files) is exit 2.
- `rote run` surfaces bundled dev UIs: the temporal runner starts its
  dev server with `ui=True` on an explicit `find_free_port()` port
  (`ui_port=None` picks one we couldn't print) and the inngest runner
  prints the dev-server dashboard URL — both live only while the run
  lasts.
- Cloud-side compilation (`rote.cloud_compile` + `_cmd_compile_cloud`):
  logged in + `--cloud`, a cloudflare runtime, or `deploy: rote-cloud`
  → the compilation runs ON rote cloud — bundle
  sync with per-file sha diff-skip, start-or-attach (an active
  compilation for the same skill attaches instead of double-spending;
  Ctrl-C cancels a run we started but only detaches from one we
  attached to), server events streamed through the same progress
  renderer/JSONL sink as local, artifacts downloaded into the standard
  `--out` layout. `--local`/`--cloud` flags (mutually exclusive);
  local-only flags (`--agent`, `--backend`, `--baseline*`, `--yes`,
  `--no-eval`) reject in cloud mode with a `--local` pointer. Config
  interplay: a config-pinned local runtime or `deploy: none` opts out
  of the cloud default (one-line reason printed); config `agent`/
  `model` NEVER leak into cloud mode — the server owns driver choice
  and validates models against its own lineup. Server contract lives
  in rote-cloud (`/v1/skills` + bundle endpoints, `/v1/skills/:id/
  compilations`, container job) — the server must be deployed before a
  CLI release that enables this. E2E-proven against local dev
  (including a drained-tenant "insufficient credits" path).
- The compile default is the compiled skill (`DEFAULT_COMPILE_RUNTIME =
  "skill"`, `builtin_compile_runtime` in cli.py, which `rote config`
  reports through too), logged in or not: no flag, env or config runtime
  → a local `skill` compile, plus one stderr line naming the default
  (and `--cloud` when logged in), and the summary's `next:` lines name
  `rote skill install` and `rote try`. The one exception is `deploy:
  rote-cloud`, which makes `cloudflare` the built-in (logged out, it exits
  2 before the agent). A local cloudflare compile while logged in still
  auto-deploys (`--no-deploy` opts out; a deploy failure exits 1 but keeps
  the local artifacts and prints the `rote deploy … --target rote-cloud`
  retry). Never a prompt (CI-safe). The parser's `--runtime` default is
  `None` — resolution happens at the top of `_cmd_compile`; don't
  reintroduce a static default in the parser.
- `rote register` + `rote serve` (compiled pipelines as MCP tools,
  FastMCP 4.x, stdio + Streamable HTTP — see
  [`docs/mcp-trigger.md`](docs/mcp-trigger.md))
- `rote eval` + the auto-emitted `compiled/scorecard.md` (static
  before/after estimate of speed, cost, determinism; live-fetched
  prices — see the eval gotchas below)
- `rote eval --run` (empirical mode: real trials of both sides —
  `claude -p` for the skill, the emitted python/DBOS app for the
  pipeline with cross-process gate signaling via `DBOSClient`; real
  judge usage captured through the emitted `$ROTE_USAGE_LOG` hook;
  measurements appended to `~/.local/share/rote/eval-corpus.jsonl`,
  or `ROTE_EVAL_CORPUS_PATH`)
- 1228 tests (1201 fast + 27 slow). Run with `pytest tests/` (fast
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
- **Not the rubric.** The compiler's rubric lives in
  [`skills/rote-compile/references/`](skills/rote-compile/references/) —
  that's what the agent reads. This CLAUDE.md is what *you* read.

If information in this file contradicts information in `ir.py`, the
test suite, or `scripts/sanity-check.sh`, **those are the source of
truth and this file is out of date.** Please fix it.
