# Parallel per-node generation for System-1 compiles

Status: the default text step of a System-1 compile with the `claude` and
`api` drivers. `codex` and `openai-api` keep the single SYSTEM1.md session,
and so does a `--backend api` compile, whose external_call modules are
vendor calls researched with web tools the parallel calls do not have.

After System-1 has decided the structure, one agent session used to write
all the text: every extracted module, every test, the judge prompts and
`system1/text.yaml`. This replaces that session with small, schema-locked
calls that run at the same time, one or two per node.

## Result

Measured 2026-10-20 with `rote bench`, 3 compiles per cell, wheels of both
commits, subscription lane (`claude -p`), concurrency 2. Serial is main at
e06c232 (the session on Sonnet 4.6, effort low). Parallel is this branch at
500238f (Sonnet 5.5 for every call, effort low). Mean, with the range.

| skill | path | fidelity | wall (s) | cost ($) |
|---|---|---|---|---|
| expense-audit | serial | 0.92 (0.90 to 0.92) | 321 (281 to 376) | 0.69 (0.60 to 0.81) |
| expense-audit | parallel | **0.95** (0.95 to 0.95) | **76** (67 to 83) | **0.48** (0.40 to 0.55) |
| ops-report | serial | 0.60 (0.50 to 0.80) | 209 (194 to 221) | 0.57 (0.47 to 0.64) |
| ops-report | parallel | **0.96** (0.96 to 0.96) | **52** (46 to 57) | **0.30** (0.27 to 0.33) |
| customer-onboarding (holdout) | serial | n/a | 188 (108 to 246) | 0.45 (0.29 to 0.59) |
| customer-onboarding (holdout) | parallel | n/a | **35** (32 to 38) | **0.15** (0.13 to 0.19) |
| saas-spend-audit (holdout) | serial | n/a | 336 (185 to 412) | 0.74 (0.42 to 0.99) |
| saas-spend-audit (holdout) | parallel | n/a | **51** (42 to 60) | **0.27** (0.25 to 0.31) |
| incident-triage (holdout) | serial | n/a | 336 (289 to 407) | 0.84 (0.65 to 1.16) |
| incident-triage (holdout) | parallel | n/a | **62** (60 to 64) | **0.44** (0.39 to 0.51) |

The held-out skills have no fidelity cases, so they are judged by
verification. Every compile in both columns verified with its tests passing,
except incident-triage, which exits 1 on both paths for the same adapter
defect (the emitted `signatures/_rote_inference.py` imports
`extracted._rote_mcp`, which the dbos emit does not ship). Structural
scores are identical across the two paths, because structure is decided
before either runs.

On the API lane (`--agent api`, the same branch plus effort on the API
call), one compile each: expense-audit 0.94 in 66 s, ops-report 0.96 in
61 s, about $0.37 and $0.29 at list prices from the token counts.

Fidelity also stopped swinging. The serial path's ops-report scored 0.50,
0.80 and 0.50. The parallel path scored 0.96 three times. The two fixes
behind that are below (the shared rule tables and unverified tool results).

Parallel compiles write fewer tests: 16 to 55 against 28 to 72 for the same
skills. Every implemented function is still tested (verification requires
it), but the depth is lower. That is an open item, not a measured fidelity
cost.

## Where the serial compile spent its time

Eleven System-1 compiles (`progress.jsonl`, the `modelUsage` totals in each
run's summary line), before this change:

| run | wall | until first module | modules | tests | repair | turns | output tok | $ |
|---|---|---|---|---|---|---|---|---|
| expense-audit (A) | 165 s | 20 | 50 | 58 | 22 | 10 | 14,719 | 0.49 |
| expense-audit (C) | 273 s | 112 | 60 | 75 | 8 | 7 | 23,287 | 0.62 |
| ops-report (A) | 184 s | 35 | 71 | 48 | 15 | 7 | 17,018 | 0.46 |
| ops-report (C) | 232 s | 78 | 72 | 67 | 6 | 7 | 20,127 | 0.52 |
| deal-monitor | 347 s | 91 | 118 | 76 | 52 | 33 | 25,551 | 0.99 |
| bdr-outreach | 260 s | 118 | 55 | 41 | 12 | 8 | 20,728 | 0.63 |
| customer-onboarding | 274 s | 91 | 60 | - | 52 | 14 | 21,411 | 0.88 |

1. **Output tokens were the wall time.** 15k to 26k output tokens, streamed
   at about 100 to 130 tokens per second, is 130 to 220 s, and the modules
   and tests came one after another in two turns.
2. **Output tokens were also about half the cost.** On expense-audit (A),
   output was $0.22 (45%), cache writes $0.17 (35%) and cache reads $0.10
   (20%). The cache reads grew with every turn, because each turn re-reads
   the whole conversation (deal-monitor read 1.4M cached tokens).
3. **Repair re-read everything.** A repair round resumed the whole
   conversation to fix one module: 7 to 52 s.
4. **Some text was not text.** In ops-report, 4 of 9 modules are MCP-bound
   `external_call` stubs, which follow from the IR alone.

## What is independent once the skeleton is frozen

A module depends on two things its own section does not show: the structure
of each value it receives, and the shape its consumers expect back. The single
session coordinated these implicitly. Written in parallel, they must be
fixed first.

- **The interface** fixes each producing node's return value as a JSON
  Schema (plus a description and, for a loop the skill gives no stop, a
  termination). It is one call per node, all at once: a node describes only
  what it produces, its consumers read what it says, so producer and
  consumer agree by construction. Each call sees the whole pipeline's
  outline and its consumers' sections. A first version asked for every
  node in one call. It took 56 s and 8.7k output tokens, against 14 s for
  the per-node calls, at the same fidelity.
- **The pieces** are then independent: one call per `pure_function`
  (module and tests together) and one per judge (its prompt). Each sees
  its section, its exact `def` line, the schema and sample value of each
  parameter, its constants, its output schema, its consumers, the skill
  text (for verbatim citations) and the rule tables.

Code writes everything that needs no model: each `external_call`'s
direct-API stub, each judge's typed input schema and `signature_spec`, each
`def` line, and the merge into `system1/text.yaml`, which now carries every
node's `output_schema` into the IR.

## The pipeline

```
skeleton (rules)              decided before this step
  -> interface                one call per node, in parallel
  -> pieces                   one call per pure_function and judge, in parallel
       each answer is checked the moment it arrives, cheapest check first:
       parses on 3.11, exact signature, stdlib imports, the test imports
       only names the module defines, the prompt's variables resolve,
       citations and outcome markers (rote.compiler.provenance), and its
       own tests in the sandbox. A failing piece is re-asked alone, with
       the problems and its files as text, and answers with edits.
  -> merge (code)             files, text.yaml, settle()
  -> verify_deliverable       contracts, tests, the prefix on the sample, dry emit
  -> per-node repair          only the nodes a finding names, at most 2 rounds
```

Checking each piece alone moved most repairs off the critical path: on the
first full version they waited for the slowest piece and a whole
verification round, and the compile took 183 s. With per-piece checks the
same skill took 64 s, and most compiles in the bench needed no
verification round at all (1 of 15).

The prefix (`rote.compiler.prefix`) now checks each result on the sample
against its node's `output_schema`, and runs past a gate when the sample
answers it. Before that, a module after the gate (expense-audit's
`file_audit_report`) could raise on every case with nothing to catch it.

## Two fixes that made fidelity stable

- **The rule tables go in the shared prompt.** A module that imports
  `extracted.rule_tables` must know the exact names and nesting. With
  only the names, one compile's `file_audit_report` guessed the FX table's
  shape, raised "no FX rate for currency EUR", and scored 0.66.
- **An MCP result no recording shows is unverified.** The interface must
  infer its shape from the skill's words, and a wrong guess (an object with
  a `threads` key, for a Gmail tool that returns the list) crashes every
  run in the consumer. That was ops-report's 0.50 on both paths. Each
  consumer of such a result is now told to accept the list, an object
  wrapping it, or JSON text, and to test each. A recorded baseline result
  replaces the guess with a schema inferred from it.

## Cost and caching

Every call shares one system prompt, byte for byte: `GENERATE.md`, the
rule tables and the skill. Two things make the prompt cache serve it:

- **One answer schema for every call.** `--json-schema` becomes a forced
  tool, and tools come first in the cache prefix, so calls with different
  schemas share nothing. Measured: seven pieces with their own schema each
  wrote about 9k cache tokens and read none. `ANSWER_SCHEMA` has every
  field required, a call fills the fields its task names, and code checks
  it did. `output_schema` travels as JSON text, because the API's
  structured outputs require every object to close its properties.
- **The first call runs alone** until it finishes or 4 s pass, since a
  cache entry is readable only once its request's response has started.

## Model per piece

Sonnet 5.5 writes every piece. Haiku 5.5, measured on the same two skills:

| model | per piece | output tokens per piece | compile |
|---|---|---|---|
| Sonnet 5.5 | 10 to 36 s | 1.8k to 6.2k | 64 s, $0.44 (expense-audit) |
| Haiku 5.5 | 26 to 63 s | 7k to 17k (mostly thinking) | failed: `structured_output_retry_exhausted` |

Haiku thinks at length even at effort low, so it is slower per piece, and
one call ended without a valid answer. That failure now retries once and
then leaves only that module unwritten. At about a tenth of the cost
(about $0.10 a compile), Haiku is worth another look for the cheap
interface calls once its structured-output reliability is measured.

## Model calls

Both lanes go through the compile driver, so auth, the subscription
`ANTHROPIC_API_KEY` scrub and the API driver's gateway settings are the
ones a compile already uses.

- `ClaudeDriver.generate`: `claude -p --output-format json --json-schema`,
  the system prompt from a file, `--tools ""` and the context-isolation
  flags, the 5-minute cache TTL, the driver's effort. Usage from
  `modelUsage` (all four buckets), cost from `total_cost_usd`. An
  `is_error` envelope inside a zero exit is a failure.
- `AnthropicApiDriver.generate`: `output_config.format` with the answer
  schema (current models reject a forced `tool_choice` with a 400), the
  shared prefix marked `cache_control`, `output_config.effort` (unset, the
  API's default spent 12k to 21k output tokens per piece against 2k to 3k
  at low), all four usage buckets, no `temperature`.

Concurrency is bounded by `--max-parallel` (default 8).

## Open items

1. incident-triage fails on both paths: the dbos emit's
   `signatures/_rote_inference.py` imports `extracted._rote_mcp`, which it
   does not ship for that pipeline. An adapter defect, not this change.
2. Test depth: parallel pieces write about half as many tests.
3. The cloud lane's API driver gives the session agent live MCP tools for
   research. The parallel calls have no tools, so that context is lost
   there.
4. The per-node prefix probe from #170 (consumers of unrecorded MCP results
   run on `[]` and `""`) could run inside each piece's own check, once #170
   lands.
