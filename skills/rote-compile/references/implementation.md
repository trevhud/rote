# Writing Extracted Implementations

Phase 3 produces `extracted/*.py` modules. This file governs **how much
of each module you implement** and how to prove what you wrote is
right. The bar depends on the node kind and on what evidence you have.

## `pure_function` nodes: implement them

A `pure_function` is code the skill already describes: parse this
table, count rows where Refunded = FALSE, flag stores below 90%, render
these three sections. The skill IS the spec, so write the real
function, in every regime, with or without `probe-context/`. A compiled
pipeline whose parsers and renderers are stubs never runs without a
human writing the code the skill already spelled out, which defeats the
point of compiling it.

What "implement" means:

1. **Every rule the skill states becomes code.** Thresholds compare
   with the operator the prose implies (`>= 10`, `> 5%`, `below 90%`).
   Lift each number into the node's `constants:` and a module-level
   constant, and enforce it in code.
2. **Input you cannot see is parsed by what the skill names.** A
   pure_function that parses an MCP result whose exact format no
   `probe-context/` shows still gets written. Accept the most direct
   form the skill implies (sheet content as CSV/TSV text with a header
   row, or a list of row dicts), find columns by the header names the
   skill gives (case- and whitespace-insensitive), and raise
   `ValueError` naming the missing column when one is absent. State the
   assumed input shape in the docstring. Never return an empty result
   for input you could not parse.
3. **Stdlib only.** No third-party imports in a pure_function module.
   It runs on every runtime and in the generated tests.
4. **Python 3.11 syntax.** rote supports 3.11, and a module that only
   parses on 3.12+ cannot be imported there. Inside an f-string's `{}`:
   no backslash (`f"{'\n'.join(x)}"` is the classic: bind
   `nl = "\n"` first, or build the string outside), no comment, no
   reuse of the enclosing quote, no line break in a single-quoted
   f-string. No `type X = ...` aliases, no `def f[T]()` generics.
5. **Signature = the node's `inputs:` keys** (next section).

### When a pure_function stays a stub

Only when the logic depends on something the skill does not state: a
business fact (a threshold "as agreed with finance", a mapping table
that lives elsewhere) or an external system (the step secretly reads a
database). Write the stub so the decision is reviewable: the whole body
after the docstring is this one statement (code before it makes the
function count as implemented, and then it must pass tests):

```python
raise NotImplementedError(
    "needs the carrier-to-region mapping: the skill says 'route by region' "
    "but never lists the regions"
)
```

The message names what is missing and why. rote lists every stub
with this message under **Open questions** in the compilation report,
and rejects a pure_function stub without one. "The input format is not
shown" is not a reason (rule 2 above), and "the rule is complex" is not
a reason either.

### Every outcome cites the skill, or is a declared assumption

Implementing a rule is not the same as inventing one. A skill often
names a case without stating its outcome: a returns skill lists
"Damaged in Transit" among the return reasons, and says which returns
are refunded at once and which go to manual review, but never says which
of the two a damaged return is.
Picking one silently, then writing tests that encode the pick, produces
green tests for a rule nobody stated. A real compile did exactly that.

So every place the code decides an output value, category, severity, or
branch carries the skill text that decides it, verbatim, in a comment:

```python
# skill: "A return over $50.00 without a receipt goes to manual review"
if ret["amount"] > RECEIPT_REQUIRED_OVER and not ret["receipt"]:
    return {"route": "manual_review"}
```

How a citation is written, and what rote accepts:

- It opens with `# skill: "` and closes at the `"` that ends a comment
  line, so the quote may itself contain quotes:
  `# skill: "Returns marked "Damaged in Transit" are refunded at once"`.
  A long quote may continue on the next comment lines.
- It quotes the sentence that decides the outcome, not a word or a
  heading. Each fragment is at least 4 words of the skill's prose
  (headings do not count), copied verbatim.
- `...` elides text between fragments. The fragments must appear in
  that order within one paragraph of the skill.
- A table row is cited whole, every cell, pipes included, since one
  cell is too short to stand alone. A markdown link is cited by its
  text. Formatting marks (`**`, backticks) may be dropped:

```python
# skill: "| Wrong Item | wrong_item | 30 days | Refund once the item is back in stock. |"
WRONG_ITEM_WINDOW_DAYS = 30
# skill: "Look up each store's region in references/stores.md"
```

rote gives each assumption its own Open questions entry in the
compilation report, with the marker's text and its file and line, so the
marker carries the whole entry:

```python
# rote: assumption: a damaged return goes to manual review, since the carrier must be billed; the skill lists "Damaged in Transit" but not whether it is refunded at once
```

Every outcome literal the code writes needs a marker: a label it
returns (alone or in a tuple), or a string it puts under a key, keyword,
or variable whose name says outcome (`severity`, `status`, `category`,
`level`, `alert`, `route`, `tier`, `decision`, `flag`, `label`,
`priority`...). A label reached by reference needs its marker where it is
defined: a module constant (`BLOCK = "block"`), a policy table
(`SEVERITY_BY_RULE = {...}`, marked on the entry or above the table), or
an Enum member.
A marker covers the statement it is on, or sits in the comment block
directly above that statement, or directly above the `if` that decides
it. One comment above an `if`/`elif` chain, or above a ladder of `if`s
that each `return`, covers the whole chain. It does not cover the next,
independent check: give each decision its own marker. rote reports an outcome literal
with no marker. When the skill does NOT state the outcome, choose one of
two:

1. **Decide, and declare it.** Write the branch with the outcome you
   think is most defensible, mark it
   `# rote: assumption: <the outcome you chose, and what the skill leaves open>`.
   rote lists it under **Open questions** in the compilation report. A
   test that pins the assumed outcome carries the same comment, so it
   reads as an assumption, not a rule. Declaring is the point:
   assumptions never fail the compile, however many there are; rote
   lists each one for review. A `# skill:` quote the
   skill does not contain does fail the compile, so when the skill is
   silent, assume and declare rather than cite.
2. **Stub the function**, when the unstated fact is what the function is
   for (a whole mapping table, a threshold that drives every branch).

Never write a `# skill:` citation for text the skill does not contain.

## `external_call` nodes: the two regimes

An `external_call` talks to a vendor API. With an `mcp:` binding the
default backend already emits a working call, and the `impl` is the
direct-API alternative (`--backend api`).

### Regime 1 — No ground truth (default)

No `probe-context/` directory exists in your work dir, and you have no
web research tools. Emit **contract-documented stubs** for the vendor
calls: the function signature, a docstring stating the input/output
contract as precisely as the skill text allows, and a
`raise NotImplementedError("<which API, and why it is a stub>")` body.
Do NOT write vendor API calls from memory — SDK and endpoint shapes
drift faster than any model's training data, and a plausible-but-wrong
implementation is worse than an honest stub because it fails at
runtime instead of at review. This regime never applies to a
pure_function.

### Regime 2 — Ground truth available (implement for real)

`probe-context/` exists (observed MCP traffic + inferred schemas from
an instrumented run of this exact skill), and/or you have WebSearch /
WebFetch. Now you are expected to write **working implementations**:

1. **Contract from observation.** The observed input payload keys are
   your function parameters; the observed result shape is your return
   contract. If the skill prose disagrees with the observed traffic,
   trust the traffic and mark the discrepancy `# rote: assumption:`.
2. **Endpoint from current docs, never memory.** Before writing a
   vendor call, use WebSearch/WebFetch to confirm the current endpoint,
   auth scheme, request/response shape, and official SDK name. Cite the
   doc URL in the module docstring. If you cannot verify a shape, fall
   back to Regime 1 for that function and say why in its stub message.
3. **Credentials via environment variables only.** One variable per
   vendor, named `<VENDOR>_API_KEY` (or the vendor's own documented
   convention, e.g. `GOOGLE_APPLICATION_CREDENTIALS`). Read it at call
   time, and raise a clear `RuntimeError` naming the variable when it
   is missing. NEVER hardcode a credential, NEVER invent a
   plausible-looking value, NEVER default to a dummy that would send a
   request with fake auth.
4. **Minimal dependencies.** Prefer the vendor's official Python SDK
   when one exists and is current; otherwise plain `httpx`. Every
   third-party dependency you introduce must be listed in the module
   docstring so the adapter's requirements emission can pick it up.
5. **Honest error surfaces.** No silent fallbacks, no bare excepts, no
   returning empty results on failure. A 4xx/5xx should raise with the
   response body attached — errors are how the durable runtime knows
   to retry or park.

## Tests are part of the deliverable

Every function you implement gets tests, in `tests/test_<module>.py`
in your work dir (one file per extracted module). Each test file imports
what it tests as `from extracted.<module> import <fn>`: rote runs
`python -m pytest tests` from the work dir root, which puts that root on
`sys.path`, so no conftest is needed.

For a **pure_function**, derive the cases from the skill itself:

- **One test per stated rule, on both sides of every boundary.** "Flag
  any store below 90% fill rate" is two cases: 89.9 flagged, 90.0 not.
  "Returns >= 10 (Warning), >= 20 (Alert)" is 9, 10, 19,
  20. Off-by-one at a threshold is the most likely bug, and the
  only one a reviewer cannot spot by reading.
- **The skill's own examples, verbatim.** A sample row, a sample
  report line, a named store (`STORE-04`), a reason code ("Damaged in
  Transit"): build the fixture from them.
- **The empty and the malformed input.** No rows, and a table missing a
  column the skill names (expect the `ValueError`).
- **A renderer's output contains what the skill requires.** Every
  section heading, KPI, and alert label the skill lists appears in the
  rendered text for a fixture that should produce it, and the alert
  labels are absent for one that should not.

For a **Regime 2** implementation, add:

- **Golden-fixture tests from observed payloads.** Every parse /
  transform / filter function gets called with a real observed payload
  (copy it from `probe-context/observed-tools.json` into the test as a
  fixture literal) and asserted against the real result. These are the
  highest-value tests you can write: they prove the implementation
  handles what production actually sends.
- **Network functions get a transport-stubbed test.** Monkeypatch the
  HTTP layer (or inject a fake client) and assert the function builds
  the documented request and parses the documented response — using
  the observed payloads as the canned response body. **No test may
  touch the network**, require credentials, or be nondeterministic:
  they must pass with plain `pytest` in an offline environment.
- **Constants get pinned.** Every constant lifted in Phase 3 (batch
  sizes, thresholds, day windows) gets one assertion, so a silent edit
  fails a test instead of shipping.

Tests use plain `pytest` and the stdlib, nothing else. If a test would
need something you can't provide deterministically, don't write a flaky
approximation. Mark the gap `# rote: assumption:` in the module instead.

## What rote checks after you stop

You have no shell, so you cannot run the tests yourself. rote runs
these checks on your work dir when you end your turn:

1. `pipeline.yaml` validates.
2. Every module an `impl:` names parses on Python 3.11, defines the
   function, and accepts the node's `inputs:` as keyword arguments.
3. Every `pure_function` is implemented, or is a stub whose
   `NotImplementedError` message says why.
4. Every `# skill: "..."` citation quotes the skill's prose (4+ words
   per fragment, in order, one paragraph), every outcome literal in a
   pure_function module has a marker.
5. Every implemented `pure_function` is called from a file under
   `tests/`.
6. `python -m pytest tests` passes. It runs in a sandbox with no
   network, no credentials, an empty HOME, and nothing readable outside
   the work dir, so a test that needs any of those is broken.

If any check fails, the problems come back to you as the next message,
verbatim, with the pytest output. Fix exactly what they name and end
your turn again. When a test and the implementation disagree, re-read
the skill: it decides which one is wrong. Never delete a test or loosen
an assertion that encodes one of the skill's rules to get to green.

## Parameters are the node's `inputs:` keys

Every adapter calls an impl as `fn(**payload)`, where the payload keys
are exactly the node's `inputs:` keys. So the function's parameter
names must be those keys, in every regime, stub or not. On a
`fan_out` node the element param carries ONE whole item, so the
function takes that item as one parameter and reads its fields:

```python
# WRONG (a real deal-monitor compile): the node binds
# `thread: match_threads_to_opps.output.matched_opp_threads`, but the
# function asks for the element's fields. Every call raises TypeError.
def fetch_thread_content(thread_id: str, thread_url: str) -> dict: ...


# RIGHT: one parameter per `inputs:` key. Fields come from the element.
def fetch_thread_content(thread: dict) -> dict:
    thread_id = thread["thread_id"]
    ...
```

`rote compile` reports a mismatch as a `contract: ERROR` line naming
the missing and unexpected parameters. Judges follow the same rule
through their input schema (`llm-judge-extraction.md`).

## What NOT to do, in any regime

- Do not write to `signatures/` from this file's rules — judge modules
  are governed by `llm-judge-extraction.md`.
- Do not call MCP at runtime from extracted code. The `mcp:` binding on
  the node (see `ir-schema.md`) is how the MCP backend wires itself;
  your `extracted/` implementation is the *direct-API* backend the
  user selects with `--backend api`. Keep both true to the same
  contract.
- Do not invent data the observation doesn't show. An inferred schema
  from one sample marks fields it saw; absence of a field is not
  license to add speculative ones.
- Do not soften a failing contract to make a test pass. The observed
  payload is the spec; if your implementation can't reproduce it,
  the implementation is wrong.

## BDR worked example

`enrich_contact_batch` under Regime 2, with an observed
`zoominfo_enrich_contacts` call in probe-context:

- Function takes `contacts: list[dict]` (the observed input key),
  chunks by the lifted `BATCH_SIZE = 10`, calls the documented
  ZoomInfo enrich endpoint with `ZOOMINFO_API_KEY` from the env, and
  returns the observed response shape (`{"contacts": [...]}`)
  normalized per the inferred schema.
- `tests/test_extracted.py` asserts: chunking math at the boundary
  (10, 11, 20 contacts), request shape against a stubbed transport
  using the observed input, response parsing against the observed
  result payload verbatim, and `BATCH_SIZE == 10`.
- Module docstring cites the ZoomInfo enrich doc URL fetched during
  implementation and lists `httpx` as the dependency.

## Pure-function worked example

A returns skill says of its Returns Log sheet: "Count all rows where
Refunded = FALSE. Break down by store (STORE-04, STORE-11, etc.) and
reason (Damaged in Transit, Wrong Item). Report the total open count."
There is no probe-context and no format sample. It is still implemented:

- `parse_returns_log(raw: str) -> dict` reads `raw` as CSV text with a
  header row, locates the `Refunded`, `Store` and `Reason` columns by
  name (case- and whitespace-insensitive), counts rows whose Refunded
  cell is `FALSE` (case-insensitive), and returns
  `{"total_open": int, "by_store": {...}, "by_reason": {...}}`. A
  missing column raises `ValueError("Returns Log has no 'Refunded'
  column; found [...]")`.
- `tests/test_parsers.py`: a fixture with STORE-04 and STORE-11 rows
  of both reasons, one of them refunded, asserting the exact totals and
  breakdowns, the `TRUE`/`false`/`FALSE` spellings, a header-only
  table (zero open), and a table without `Refunded` (the `ValueError`).
