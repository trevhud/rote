# Parallel per-node generation for System-1 compiles

Status: design, then built behind the default System-1 path (see the
measured table at the end once it exists).

After System-1 has decided the structure, one agent session writes all the
text: every extracted module, every test, the judge prompts and
`system1/text.yaml`. This doc replaces that session with small,
schema-locked generations that run at the same time, one per node.

## Where a System-1 compile spends its time today

Eleven System-1 compiles from the last two days (`progress.jsonl`, the
`modelUsage` totals in each run's summary line), default driver
(`claude -p`, Sonnet 4.6, effort low):

| run | wall | until first module | modules | tests | repair | turns | output tok | cache write | cache read | $ |
|---|---|---|---|---|---|---|---|---|---|---|
| expense-audit (A) | 165 s | 20 | 50 | 58 | 22 | 10 | 14,719 | 45,195 | 320,284 | 0.49 |
| expense-audit (C) | 273 s | 112 | 60 | 75 | 8 | 7 | 23,287 | 54,158 | 214,341 | 0.62 |
| expense-audit (rules) | 220 s | 20 | 79 | 70 | 31 | 22 | 18,449 | 49,309 | 771,371 | 0.69 |
| ops-report (A) | 184 s | 35 | 71 | 48 | 15 | 7 | 17,018 | 38,948 | 195,526 | 0.46 |
| ops-report (C) | 232 s | 78 | 72 | 67 | 6 | 7 | 20,127 | 41,174 | 207,410 | 0.52 |
| ops-report (rules) | 205 s | 69 | 63 | 51 | 7 | 7 | 16,947 | 37,723 | 199,113 | 0.46 |
| ops-report (clef) | 218 s | 76 | 80 | 46 | 0 | 18 | 17,677 | 37,604 | 587,435 | 0.58 |
| ops-report (haiku tie-break) | 205 s | 46 | 69 | 66 | 14 | 23 | 16,461 | 37,951 | 767,225 | 0.62 |
| deal-monitor | 347 s | 91 | 118 | 76 | 52 | 33 | 25,551 | 49,820 | 1,385,326 | 0.99 |
| bdr-outreach | 260 s | 118 | 55 | 41 | 12 | 8 | 20,728 | 63,391 | 271,981 | 0.63 |
| customer-onboarding | 274 s | 91 | 60 | - | 52 | 14 | 21,411 | 59,120 | 673,882 | 0.88 |

"Until first module" is reading plus generating the first module, since a
Write appears when its tool call has streamed. "Modules" runs from the first
module Write to the last, and "tests" from there to the last test Write.

What the numbers say:

1. **Output tokens are the wall time.** 15k to 26k output tokens per
   compile, streamed at roughly 100 to 130 tokens per second, is 130 to
   220 s. Modules and tests are written one after another in two turns,
   so the whole compile waits on one stream.
2. **Output tokens are also about half the cost.** On expense-audit (A):
   output $0.22 (45%), cache writes $0.17 (35%), cache reads $0.10 (20%).
   The cache reads grow with every turn, because each turn re-reads the
   whole conversation (deal-monitor read 1.4M cached tokens in 33 turns).
3. **Repair re-reads everything.** A repair round resumes the full
   conversation to fix one module: 7 to 52 s and a growing cache read.
4. **Some of the text is not text.** In ops-report, 4 of 9 modules are
   MCP-bound `external_call` stubs (`raise NotImplementedError`, the
   `--backend api` alternative with no probe context). Their content
   follows from the IR alone.

## What is independent once the skeleton is frozen

Every node's module depends on two things it cannot see from its own
section: the shape of each value it receives, and the shape its consumers
expect it to return. The single session coordinates these implicitly (it
wrote the producer a turn earlier). Generated in parallel, they must be
fixed first, or node B reads `result["threads"]` from a node A that
returns a list. That exact defect shipped in a serial compile
(ops-report, #162).

So the work splits in two:

- **The interface** (one call, needs the whole skill): every producing
  node's return value as a JSON Schema, a one-sentence description per
  node, and a stop condition for any loop the skill gives none. Small
  output, about 1 to 3k tokens.
- **The pieces** (one call each, all at once): given the interface, each
  `pure_function` module and its test, and each judge's prompt, are
  independent of each other. Each needs only: the node's section text, its
  exact signature, the schema (and the sample value, when there is one) of
  every parameter, its constants, its output schema, which consumers read
  it, the rule-table names, and the skill text (for verbatim citations).

What needs no model at all:

- an `external_call` without probe context: a contract-documented stub
  (implementation.md, Regime 1) rendered from the IR and the interface;
- each judge's `signature_spec.input_schema` (its properties are exactly
  the node's `inputs:` keys, each typed by its producer's output schema)
  and `output_schema` (from the interface), and `client`;
- the function's `def` line, which the piece must use verbatim;
- `system1/text.yaml`, merged from the pieces' answers.

## The pipeline

```
skeleton (rules, no model)            already built
  -> deterministic pieces              stubs, signatures, judge schemas
  -> interface call (1)                output schemas, descriptions, loop stops
  -> validate interface (code)         fan_out sources return {items: [...]}, etc.
  -> pieces (N, bounded concurrency)   module + test per node, prompt per judge
       each answer checked by code at once (parses on 3.11, exact signature,
       stdlib imports, test imports the function, prompt variables resolve);
       a failing piece is re-asked alone, without waiting for the others
  -> merge (code)                      files, text.yaml, settle()
  -> verify_deliverable (existing)     tests, contracts, provenance, prefix, dry emit
  -> per-node repair (parallel)        only the nodes a finding names, max 2 rounds
```

Checks run in the order the user asked for: schema, then code, then models.
The interface call and every piece are schema-locked (`claude -p
--json-schema`, a forced tool call; the API driver forces a tool the same
way), and the interface's response schema names every producing node as a
required key, so a node cannot be skipped or invented. Code checks each
answer before anything runs. Only what survives both reaches verification.

The prefix run (`rote.compiler.prefix`) already executes the deterministic
nodes on the sample input. It also checks each result against the node's
interface schema, so a module that returns a `set`, or a dict with the
wrong keys, is caught on data no model wrote.

## Cost and caching

Every call shares one system prompt: the piece rubric (`GENERATE.md`) and
the skill's text, identical bytes for every call in a compile. The
interface call runs first and writes that prefix to the prompt cache; the
pieces start after it and read it at a tenth of the input price. A probe
(two pieces, 10k-token shared prefix) read 10,042 of 10,428 prefix tokens
from cache when the prefix had been written first, and wrote all of it
twice when the two started together.

Per piece, measured on two expense-audit nodes (`claude -p`, effort low):

| model | wall | output tok (thinking) | $ |
|---|---|---|---|
| Sonnet 5.5 | 12 to 17 s | 1.9k to 2.0k (0.2k to 0.4k) | 0.06 cold, about 0.03 warm |
| Haiku 5.5 | 23 to 33 s | 5.7k to 8.3k (3.3k to 5.6k) | 0.003 to 0.004 |

Haiku 5.5 is 15x cheaper per piece but slower, because it thinks at length
even at effort low. Which model writes which piece is decided by the
fidelity bench below, not by these two samples.

## Model calls

Both lanes go through the compile driver, so auth, the subscription
`ANTHROPIC_API_KEY` scrub and the API driver's gateway settings are the
ones a compile already uses:

- `ClaudeDriver.generate`: `claude -p --output-format json --json-schema`,
  the system prompt from a file, `--tools ""` and the context-isolation
  flags (no CLAUDE.md, no settings, no MCP), the 5-minute cache TTL, the
  effort setting. Usage from `modelUsage` (all four buckets), cost from
  `total_cost_usd`. An `is_error` envelope inside a zero exit is a
  failure.
- `AnthropicApiDriver.generate`: `messages.create` with one forced tool
  whose `input_schema` is the response schema, the shared prefix marked
  `cache_control`, all four usage buckets. No `temperature`.

A driver without `generate` (codex, openai-api) keeps the single session.

Concurrency is bounded by `--max-parallel` (default 8).

## What could make it worse, and the check for each

| risk | check |
|---|---|
| a piece misreads an upstream value | the interface schema in its card; the prefix checks results against it on the sample |
| two pieces disagree on a field name | only the interface names fields; every piece reads the same schema |
| a test pins behavior its module does not have | the same piece writes both; repair gets both and the skill decides |
| a repair breaks a neighbor | a repair changes only its node's two files; the next verification runs everything |
| the interface is wrong for a node | its consumers fail on the sample in the prefix; repair can then return a module that matches the schema, or (round 2) the interface entry is re-asked |
