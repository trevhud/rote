# Changelog

All notable changes to `rote` (distributed on PyPI as
[`rote-cli`](https://pypi.org/project/rote-cli/)) are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While `rote` is pre-1.0, minor versions may include breaking changes.

## [Unreleased]

### Added

- **`--tiebreak haiku` (the default): Claude Haiku decides only the steps
  System-1 is unsure of.** One schema-locked `claude-haiku-5-5` call a
  compile, billed like the compile (subscription under the claude driver,
  the api driver's own client under `--agent api`), about $0.001 and 6 s.
  On the held-out skills it lifts kind accuracy from 0.49 to 0.68 with the
  rules, 0.66 to 0.74 with Clef and 0.67 to 0.72 with Jev. `--tiebreak
  none` (or `tiebreak: none`, `ROTE_TIEBREAK`) turns it off. With no lane
  the compile fails and says so. `rote bench`'s `cost$` now includes
  System-1's decision calls.

### Changed

- **System-1 is the default `rote compile` path.** rote decides the pipeline
  structure with rules before the agent runs, and the agent writes only text.
  On the bench, a default compile of expense-audit or ops-report costs about
  $0.47 and takes about 3 minutes, against about $1.00 and 6 to 7 minutes on
  the full-rubric path.
  `--no-system1` runs the full-rubric agent. `--update` always runs it,
  because System-1 does not support incremental re-compilation yet, and the
  compile log says so. The `--system1` flag is gone: there is nothing to opt
  into. `rote analyze` takes the same flags. In cloud mode `--no-system1`
  and `--decider` are local-only flags. `Compiler`
  defaults to the same rule (`system1=None`), so library callers such as
  rote-cloud get System-1 too.

### Added

- **Effort knob for Anthropic judges and agent loops.** `ROTE_EFFORT_<NODE_ID>`
  sets one node's effort, and `ROTE_JUDGE_EFFORT` sets it for every node. An
  empty value omits it. The SDK lanes send `output_config.effort`; the
  subscription lane passes `claude -p --effort`, and an unknown level fails
  there too instead of being ignored. Unset, judges still send no effort (the
  API runs `high`): measured on 8 example judges, `low` saved 15 to 33% of
  output tokens but changed decisions. Works on every runtime.

### Fixed

- **An unknown effort level fails before a subscription call is paid for.**
  `claude -p` ignores an unknown `--effort` with only a warning, so rote now
  asks the CLI with a free `claude --effort <level> --version` first, and
  self-checks once per process that the CLI still warns in a way rote
  recognizes.

- **`ROTE_TEMPERATURE_<NODE_ID>` reaches judges on DBOS-TS and Inngest.**
  Those runtimes pass each judge a narrow env object, which left the
  temperature knob out, so an empty value could not drop the parameter.
- **Emitted judges work on claude-sonnet-5-5.** That model rejects a forced
  `tool_choice` with HTTP 400, and every Anthropic judge forced one, so
  pointing `ROTE_MODEL_<ID>` at it failed every judge call. Judges on every
  Anthropic lane (Python and TypeScript) now use structured outputs
  (`output_config.format`), on every model. The request schema is rewritten
  into the subset structured outputs accepts (objects closed, `oneOf` as
  `anyOf`, bounds moved into descriptions). The judge still validates the
  answer against the original. A `refusal` or `max_tokens` stop is an
  error, never data. An anthropic judge whose output schema has an object that
  admits undeclared keys (no properties, or `additionalProperties: true`
  or a schema) or an untyped value now fails at `rote emit` and in the
  compile's dry emit, naming the path. The dbos README and a new temporal
  `requirements.txt` pin `anthropic>=1.15`, the SDK floor with
  `output_config`. An
  agent loop's one extra turn after it ends without `submit_result` uses
  the same mechanism.
- **System-1 binds MCP arguments only from schema names that pass grounding's
  filter.** It read raw property names from a server's tool schema, so a name
  that is not identifier-shaped became an `mcp.args` key and the skeleton failed
  IR validation: a hostile or sloppy server could break the compile. Such names
  are now withheld and listed under Open questions, the same as in the agent's
  brief.
- **`mcp.args: {}` now calls the tool with no arguments, on every runtime.** It
  used to behave like an omitted `args` and send the step's whole payload as the
  tool's arguments. Only an omitted (null) `args` passes the payload through. A
  hand-written pipeline that relied on `args: {}` as a pass-through must drop the
  key; none of the examples does.
- **System-1: an MCP step's arguments come from the step, never the whole input.**
  The skeleton binds the parameters the step states (`Search query: ...`,
  `File ID: ...`, `Tab: ...`) as constants in `mcp.args`, named by the tool's input
  schema when grounding has one, and calls a tool with no arguments rather than
  forwarding the pipeline input when the step states none.
- **System-1: a write to no named system is the pipeline's output.** "Save it to
  the shared folder" no longer becomes an external_call stub that raises
  "target system not specified"; a named destination (HubSpot, #channel, a URL,
  a tool, a UI action) still does.

### Changed
- **`rote emit` ships every top-level `.py` in `compiled/extracted/`,** not only
  the modules a node's `impl:` names, for the dbos and python runtimes. A shared
  helper the agent wrote (`extracted/_common.py`) used to stay behind, and the
  emitted app crashed on its first import. A scratch `.py` left in that folder
  now ships too, and from there to a deploy. Subdirectories, `__pycache__` and
  non-`.py` files do not. Verification's dry emit also flags any `extracted`
  import the emitted runtime cannot satisfy.

### Added
- **`--decider jev`: Typesafe's Jev decision model as a System-1 decider,
  beside Clef.** Both run through one Workers AI client with their own
  route and frozen thresholds. Jev's thresholds were fitted on the five dev
  skills and then measured once on the held-out set: kind 0.67 against the
  rules' 0.49, 3 of 3 gates against 2 of 3, one false gate, about $0.001 a
  compile. Clef there: kind 0.66, 3 of 3 gates, no false gate, about $0.005.
  The built-in default stays `rules`. Not yet verified live: an end-to-end
  compile with `--decider jev`, and Jev's calls tagged with
  `CLOUDFLARE_AI_GATEWAY_METADATA` in the gateway logs. Jev's own calls ran
  live in the skeleton measurement, and the shared metadata code is proven
  live through Clef.
- **The decider is a layered setting.** `--decider`, `ROTE_DECIDER`, or `decider:` in `rote.yaml` or the user
  config, with strict validation. `rote init` asks for it and `rote config`
  shows it. A model decider without `CLOUDFLARE_ACCOUNT_ID` and
  `CLOUDFLARE_API_TOKEN` fails the compile. A `--decider` flag on a compile
  that does not run System-1 (`--no-system1`, `--update`) is a usage error.
  A configured one is skipped for that compile, with a one-line note.
- **`rote bench --fidelity`: does the compiled pipeline behave like the
  skill?** Each compiled pipeline runs on fixed cases (input, gate answers,
  fixture MCP responses served by a local fastmcp server) and is scored
  against the skill itself run as an agent on the same cases, several
  times, counting only what every agent run agrees on: outputs matched by
  value, outcome, reads, ordered side effects, and gate payload fit.
  References are recorded once with `--record-reference`, and
  `--fidelity-of DIR` scores any compiled dir. Cases and references ship
  for expense-audit and ops-report. See `docs/compile-bench.md`.
- **System-1 compile: rules decide, the agent only
  writes code.** Before the agent runs, rote decides the whole pipeline
  structure from the skill's text with no inference: steps and node kinds,
  data flow, fan_out, per-item loops, mandatory flags, gates with their
  signals, payloads and timeouts, input fields named in the skill's own
  words, constants, and the skill's markdown rule tables as tested Python.
  Every decision carries a confidence and the rule behind it. The agent
  then gets a short rubric and writes only module implementations, tests
  and judge prompts. rote restores every decision after each of its stops,
  lists any change it proposes without applying it, and writes
  `compile-report.md` itself from data. It is the default path (see
  Changed). See
  `docs/design/system1-compile.md`.
- **`rote bench`, a scored compile benchmark (dev tool, spends tokens).**
  It compiles the example skills, scores each `pipeline.yaml` against the
  hand-written `expected/pipeline.yaml` (node and kind match, edges, wave
  order, fan_out, mandatory, gates, inputs), re-verifies the deliverable,
  checks the dbos and cloudflare emits, and writes a diffable
  `results.json` (`--compare`). Hidden from `rote --help`. A pre-#144
  baseline is in `examples/bench/baseline.json`. See
  `docs/compile-bench.md`.
- **`rote serve` always exposes two built-in tools.** With no registry
  file the server used to answer `tools/list` with nothing, which left
  MCP directories (Glama scores a server by its tool definitions) with
  no tool to score and a fresh install with nothing to call.
  `rote_list_pipelines` lists every registered pipeline (name,
  description, runtime, trigger summary, trigger/status/signal tool
  names, `pipeline.yaml` path, registration time) plus the registry path
  and whether the file exists, with an optional `runtime` filter. An
  empty registry returns an empty list. `rote_describe_pipeline` returns
  one entry's input schema and trigger configuration with URL passwords
  replaced by `***`, and names the registered pipelines when asked for
  an unknown one. Both are read-only, return structured content with an
  output schema, and re-read the registry per call. The `rote_` prefix
  is reserved: `rote register` refuses it, and an entry in a hand-edited
  registry whose trigger or companion name collides with a built-in is
  skipped with one stderr warning instead of shadowing it.

### Fixed
- A Cloudflare app whose judge returns free-form objects (`{type: object}`
  with no properties, alone, in an array or nested) or an untyped field
  now compiles. The judge's Zod schema typed them `Record<string,
  unknown>` and `unknown`, which `step.do`'s `Rpc.Serializable`
  constraint rejects (TS2769), so `tsc` failed on emit (found by `rote
  bench` on deal-monitor). They are now a five-level JSON value type,
  declared in a signature module only when used. DBOS-TS and Inngest
  compiled before and still do. The typed-impls e2e covers the shape on
  all three.
- A `tools/list` or tool call that landed while the registry file was
  corrupt (hand-edit typo, torn write) raised inside the provider and
  FastMCP answered with no tools. Only the background watcher tolerated
  a corrupt file. Every read path now logs the parse error and keeps
  serving the previous entries, and the next read retries, so fixing
  the file recovers the server without a restart.

## [0.15.0] - 2026-10-19

### Upgrading from 0.14
This release has breaking changes. Each one has a full entry below.
- **DBOS 3 and DBOS Transact TS v5 are required** (`dbos>=3.3,<4`,
  `@dbos-inc/dbos-sdk ^5`). DBOS 3 cannot read workflows written by 2.x:
  let in-flight 2.x workflows finish, then re-emit your DBOS apps.
- **Implementations receive their node's constants.** On every runtime,
  a node's `constants:` arrive as arguments (Python keyword arguments,
  TS input fields). `rote emit` now refuses a Python implementation that
  can't accept them and names the file to fix. Make constants that every
  caller passes required parameters with no default.
- **Agent loops return their declared output keys.** A loop now submits
  its result through a `submit_result` tool that matches the node's
  declared `output:`, and downstream nodes receive those keys. Provider
  and iteration counts moved to the usage log.
- **`anthropic>=1.15`** and **`fastmcp>=4.1.1,<5`** are the new floors,
  and `pytest` is now a runtime dependency, so compile-time verification
  works on a plain `pip`/`uv` install.

### Added
- `mcp.args` takes fixed values: `{literal: "BDR do not contact"}`, or a
  number or boolean, next to the existing references (a bare string,
  which always names a payload key or a constant). A real BDR compile
  wrote literal text there and the IR rejected it. The error now names the
  `{literal: ...}` form. Types are strict and validated in the IR, and
  every MCP runtime (dbos, dbos-ts, cloudflare, inngest) renders a literal
  as an escaped value, never spliced code. The compile rubric tells the
  agent to use literals for fixed tool arguments.
- **`rote example [<name>]`: the free first run without a clone.** pip and
  uv installs never got `examples/`, so the expense-audit run in the
  README started with `git clone`. The wheel now carries the examples
  that run with no credentials (only `expense-audit` for now), packaged
  from `examples/` by pyproject's force-include rather than copied into
  `src/`. `rote example` lists them. `rote example expense-audit` copies
  the skill, both IRs, the working `extracted/` modules and the sample
  inputs into `./expense-audit` (refusing an existing directory unless
  `--force`), then prints the emit, run and eval commands for that copy.
  `uvx --from rote-cli rote example expense-audit` works with nothing
  installed.
- **The scorecard prices Cloudflare Workflows infrastructure.** Steps
  and storage started billing on 2026-08-10. For a `cloudflare` or
  `rote-cloud` pipeline, `rote eval` and `rote compile`'s scorecard now
  add a per-run table of steps, requests, CPU and storage at the paid
  plan's marginal rates, plus how many runs a month fit in the included
  steps. Prices come live from the table in
  `cloudflare/cloudflare-docs@production:src/content/partials/workflows/workflows-pricing.mdx`
  with the model catalog's 24h cache, and the scorecard cites the file
  and fetch time. A missing row or a cell outside the known grammar is a
  loud `PricingError`, never a guessed number. Steps per run come from
  the IR, with the same fan widths as the token estimate. Gates count as
  a step (a stated, conservative assumption). CPU per step, state bytes
  per step and retention (30 days, the Workers Paid account maximum)
  are stated priors, each overridable with `--prior`. `rote eval
  --runtime` picks the runtime; a `compile --out` dir with one emitted
  runtime is detected.
- **A free first run.** `examples/expense-audit/` is a month-end expense
  audit skill with a hand-compiled pipeline whose extracted modules are
  working code, so `rote emit` + `rote run` finish end to end with no
  credentials and no tokens on python (`pipeline-draft.yaml`, no gate),
  dbos, and temporal (`pipeline.yaml`, with a `controller_review` HITL
  gate). It covers a three-node parallel entry wave, a `fan_out` audit
  per employee, mandatory checks, and policy constants lifted out of
  prose. The README opens with it as "Try it free in 60 seconds", and
  `tests/test_expense_audit_example.py` runs it on python in the fast
  suite and on dbos and temporal in the slow suite (now in CI). The
  example also carries a measured `rote eval --run` scorecard (agent
  42.9 s and $0.22 per run vs pipeline 9.8 s and $0, identical output
  only on the pipeline side) and a snapshot of a real `rote compile` of
  the skill under `runs/`, with its differences from the hand-written IR
  noted.
- **`rote mcp login` follows the MCP 2026-07-28 registration order.** A
  pre-registered client from the registry wins, then rote's Client ID
  Metadata Document (client_id `https://roteskills.com/oauth/rote-cli.json`,
  source in `docs/oauth/rote-cli.json`) when the authorization server
  advertises `client_id_metadata_document_supported`, then dynamic client
  registration. A server that offers neither (GitHub's shape) now fails
  with an error naming `rote mcp add --client-id`, instead of a POST to a
  guessed `<resource-origin>/register`. The login binds the first free
  port of a published set (33418 to 33421), because a CIMD server matches
  redirect URIs exactly. `ROTE_MCP_CLIENT_METADATA_URL` overrides the
  document URL; empty turns CIMD off. The document is offered only after
  a fetch shows it published and identical to the packaged copy; until
  then the login uses DCR as before, with a one-line notice.
- `rote mcp add --client-id` records the issuer the client belongs to
  (discovered from the server's protected-resource metadata, or
  `--issuer`), so a pre-registered client is bound from its first login.
- **Token file contract v2** adds `issuer`, the authorization server the
  credentials belong to. Readers accept v1. A pre-registered client is
  bound to the issuer it logged in with, and a later login against a
  different authorization server stops before the browser opens.
- **`RoteMcpInputRequired`.** A server that asks for input in the middle
  of a call (a 2026-07-28 `input_required` result, or `-32021` for a
  capability the client did not declare) fails the step with the server,
  tool, and requested inputs named, on all four MCP runtimes. It is not
  an auth failure: it never parks, and it skips the retry budget.
- **`/rote:implement` plugin skill.** After `rote emit` or `rote compile`,
  the Claude Code plugin can now fill in the emitted `extracted/*` stubs
  without spending compilation tokens. The skill finds the TODO list
  through `rote emit --json`'s `unimplemented_stubs`, reads each node's
  IR contract and its source SKILL.md section, writes real code in
  user-owned files only, handles `.new` siblings from re-emit, and
  verifies with `rote run`. `tests/test_plugin_skills.py` checks every
  plugin skill's frontmatter and links, runs each stub and non-stub
  example in the skill through rote's own stub detector, and emits every
  runtime to confirm that each file outside `unimplemented_stubs`
  (generated agent-loop modules included) is on the skill's do-not-edit
  list.
- **The compiler writes real code for `pure_function` nodes, with tests.**
  A parser, validator, or renderer the skill spells out is implemented,
  and `tests/test_<module>.py` checks it against cases built from the
  skill's own rules and examples. A pure_function stays a stub only when
  the skill leaves a fact unstated, and the stub's `NotImplementedError`
  message says which. Vendor calls in `external_call` modules stay stubs
  unless an `mcp:` binding (or probe ground truth) makes them real.
- **`rote compile` checks the deliverable before calling it done, and the
  agent repairs what fails.** After the agent stops, rote validates the
  IR, checks every module the IR names (it defines the function, accepts
  the node's payload, and parses on Python 3.11), requires every
  pure_function to be implemented and tested or an explained stub, and
  runs the generated tests. Problems go back into the same conversation,
  verbatim, for up to two repair rounds. The `claude` driver now repairs
  too, through `claude -p --resume`. A deliverable that still fails is
  kept, and `rote compile` exits 1 and skips the rote cloud deploy.
  The summary prints how many impl nodes were implemented and every
  stub with its reason (`verification` in `--json`).
- **Generated tests run in an OS sandbox, or not at all.** They are code
  an LLM wrote from a possibly third-party skill, so rote runs them under
  `sandbox-exec` (macOS) or bubblewrap (Linux): no network, no reads
  outside the work dir and the Python install, no exec but the
  interpreter, an env of only `PATH` and locale, and an empty throwaway
  `HOME`. With no working sandbox they are skipped, and the compile
  output and a `## rote verification` section in compile-report.md say
  the deliverable is unverified. `--trust-skill` runs them without a
  sandbox; `--no-verify-tests` never runs them (both also on
  `rote analyze`). The static checks always run.
- **Every outcome the compiled code decides cites the skill, or is a
  declared assumption.** The rubric has the agent quote the skill text
  behind each value, category, and branch (`# skill: "..."`), and mark
  an outcome the skill leaves open `# rote: assumption:`. Verification
  rejects a quote the skill does not contain, and recorded assumptions
  with an empty Open questions list in compile-report.md.
- `rote.contracts` reports syntax that parses on the running Python but
  not on 3.11 (`needs_newer_python`). `ast.parse(feature_version=...)`
  misses PEP 701 f-strings, and a real compile on 3.12 wrote a backslash
  inside an f-string expression that 3.11 cannot import.

### Changed
- CI and the release workflow's test job install from `uv.lock`
  (`uv sync --frozen --extra dev` via `astral-sh/setup-uv`) instead of
  `pip install -e ".[dev]"`, which resolved the newest versions on every
  run and so never tested the lockfile developers use. Each job also
  runs `uv lock --check`, failing when the lock is out of date with
  pyproject. The newest-dependencies workflow still covers what a fresh
  PyPI install gets. The Python e2e job also reserves the OAuth callback
  ports (33418-33421) from Linux's ephemeral range, and the registration
  tests check the first port is free before they start: one run had an
  unrelated socket on 33418 for its whole duration, which surfaced as
  nine "rejected the static client credentials" failures.
- The `claude` compiler driver runs isolated from the invoking user's
  Claude Code setup: the same context flags as the emitted judge lane
  (`--setting-sources ""`, `--strict-mcp-config`, auto memory off, now
  one shared definition), `--tools` set to its allowlist, and the work
  dir as its cwd. A user's
  prose-lint hook had cost a real compile 12 of its 48 turns in one-line
  report edits, the same settings granted Bash, and run from a repo the
  agent read that repo's earlier compiled output of the same skill.
- The compiler rubric writes `compile-report.md` once, in one Write at
  the end, and never spends a turn on a progress marker alone.
- **The DBOS runtimes target DBOS 3 (Python) and DBOS Transact v5
  (TypeScript).** The `dbos` extra now requires `dbos>=3.3,<4`. Emitted
  `main.py` no longer constructs a `Queue` (DBOS 3 raises on import);
  it declares `QUEUE_NAME`, enqueues through `DBOS.enqueue_workflow`,
  and exposes `launch()`, which runs `DBOS.launch()` and then
  `DBOS.register_queue`. Anything that runs an emitted app in-process
  must call `launch()` instead of a bare `DBOS.launch()`, or runs
  enqueued by `rote serve` stay `ENQUEUED`. The removed
  `run_admin_server` key is gone from both runtimes' config, and the
  emitted DBOS-TS `package.json` pins `@dbos-inc/dbos-sdk` `^5.3.0`.
  Supporting 2.x alongside 3.x was rejected: a 2.x process or client
  cannot read workflows 3.x wrote, so a rote CLI on 2.x would fail
  against an app installed today. Apps emitted for 2.x need a re-emit.
- `rote mcp login` / `release` now finds a parked DBOS-TS workflow's
  system database the way the v5 SDK does. v5 stopped reading
  `dbos-config.yaml` at launch, so a `system_database_url` key there no
  longer counts, and the default database name follows the SDK's
  `toDbName()` (lowercased, leading digit prefixed).
- A new `Newest dependencies` workflow installs every extra at the top
  of its range with no lock or cache, daily and on PRs that touch
  dependencies or DBOS code, and runs the fast suite plus the DBOS
  Python and DBOS-TS e2e suites. An upstream major that breaks rote now
  turns CI red instead of reaching fresh installs first.
- **fastmcp 4.** The `mcp` and `serve` extras now require
  `fastmcp>=4.1.1,<5` (MCP Python SDK v2, httpx2), replacing the
  `>=3.4.2,<4` pin. Verified against FastMCP, TypeScript SDK v1 and v2,
  and five hosted servers on both protocol eras. Emitted DBOS READMEs
  print the same range. Supersedes the blind `<5` bump in #92.
- **Emitted TS helpers speak MCP 2026-07-28.** `_roteMcp.ts` (DBOS-TS,
  Inngest) and the Workers helper move from `@modelcontextprotocol/sdk`
  ^1.29 to `@modelcontextprotocol/client` ^2.3.1 with
  `versionNegotiation: { mode: "auto" }`. v1 cannot reach a modern-only
  server at all (`-32022`), and v2's default `legacy` mode behaves the
  same, so the mode is set explicitly. A 401 is now read from
  `SdkHttpError.status`. Proven against legacy stateless, legacy
  sessionful, dual-era and modern-only TS SDK servers, from both helper
  languages, in `tests/test_mcp_era_matrix_e2e.py`.
- `rote mcp headers` (and any refresh outside `login`) never opens a
  browser or registers a client: it refreshes or fails with the login
  hint.
- **Breaking: an impl must accept its node's constants.** The Python
  runtimes (dbos, python, temporal) now pass a `pure_function` /
  `external_call` node's IR `constants` to its `impl` as keyword
  arguments, merged in the step body so every caller gets them: the DAG,
  each fan_out element, and an agent loop calling a `loop_body` sub-node
  as a tool. Before, no step received them, so nodes sharing one `impl`
  with different constants (ops-report's three fetch nodes, each with its
  own `file_id`) all got the same empty payload. An `extracted/`
  function that takes neither the constant nor `**kwargs` would raise
  TypeError mid-run, so emission now refuses it up front, naming the node
  and parameter, and `rote emit` reports it as a `constant_not_accepted`
  contract finding. Most committed `examples/*/runs/` hard-code their
  constants and no longer emit on a Python runtime until recompiled. The
  BDR and expense-audit example modules take theirs as keyword
  parameters, and expense-audit's now apply them, so editing a constant
  in its pipeline.yaml changes the rule. The refusal names the full path
  of the file to fix, and `rote compile` exits 1 with the `rote emit`
  retry command instead of a traceback.
- A constant reaches an MCP tool only when `mcp.args` names it, the same
  on every runtime. A constant that reuses an input's name is an IR
  validation error for impl-backed kinds. The stub for a shared `impl`
  lists every calling node's constants.
- **Breaking (TS): the TS runtimes pass each node's constants to its
  impl, as the Python runtimes do.** Cloudflare, Inngest and DBOS-TS add
  a `pure_function` / `external_call` node's IR constants to the impl's
  input after its inputs, on DAG calls, on each fan_out element and on an
  agent loop's loop_body tool calls (the model's tool schema never shows
  them). Before, they only appeared in the stub's doc comment, so a TS
  impl had to hard-code them and an edit in pipeline.yaml did nothing. An
  MCP-backed step still sends a tool a constant only when `mcp.args`
  names it. A typed TS impl whose input type does not declare its node's
  constants now fails `tsc` with an excess-property error; add them to
  the input type.

### Fixed
- **Assumptions no longer fail a compile.** A real bdr-outreach compile
  exited 1 over 15 declared `# rote: assumption:` markers and one
  citation that was verbatim but for its quote marks. Now only a
  citation the skill does not contain fails. Every assumption is listed
  by node and `file:line` in the compile output, compile-report.md,
  `--json` and the progress.jsonl summary. An unmarked outcome or an
  assumption missing from Open questions is still sent back for repair,
  and is a warning if it survives. `--strict-provenance` (compile and
  analyze) fails on any of it, for CI. Citations now match regardless of
  quote style (`'`, `"`, curly quotes).
- Python string literals in emitted code keep characters outside the BMP
  intact. `json.dumps` spelled one as a surrogate pair, which Python reads
  as two lone surrogates, so a constant like an emoji reached its impl or
  tool changed.
- Untrusted IR text never lands invisible in emitted source. A bidi
  override, zero-width space, BOM or other control or format character
  from a pipeline.yaml is written as an escape in Python literals,
  docstrings, TS comments and the Cloudflare README (the Trojan Source
  shape, CVE-2021-42574). A lone surrogate, which used to crash emit with
  UnicodeEncodeError partway through writing files, now emits code that
  round-trips it.
- **A failed `claude -p` compile reports what it cost, and the output cap
  no longer ends it.** A real invoice-push compile died after 28 minutes
  on "Claude's response exceeded the 32000 output token maximum" and
  reported no cost anywhere. The failure's usage and cost are now in the
  error and in progress.jsonl. The compiler's CLI child gets
  `CLAUDE_CODE_MAX_OUTPUT_TOKENS=64000` unless the caller set one, and a
  run that still hits the cap is resumed once with an instruction to
  write large files in pieces.
- **Generated tests run under a uv-managed Python.** CI's setup-uv
  interpreter lives in `~/.local/share/uv/python`, and every sandboxed
  generated-test run died with "No module named 'encodings'": the child
  was launched through the venv, whose `pyvenv.cfg` points at a uv
  minor-version symlink nothing in the bwrap namespace binds. The child
  is now launched by the resolved binary. The run's summary is also
  taken from pytest's stdout, not from a termcap warning the interpreter
  prints to stderr on exit. The sandbox tests now check the real
  property (no bind exposes the home's contents, planted secrets in the
  home and the venv root are unreadable) instead of "nothing under /home
  is bound", which failed on any in-tree venv.
- **A compile no longer declares success on an IR `rote emit` refuses.**
  A real bdr-outreach compile ($4.46) exited 1 at emit because
  `email_template_loop` took its own `loop_body` sub-node
  `personalize_email` as an input, a rule only the adapters checked.
  Input reachability (no same-wave or later source, no loop-body
  sub-node) and cycles are now IR validation errors, so the repair loop
  sends them back. Verification also dry-emits for the compile's
  `--runtime` and sends any refusal back (`emit_refused`), which covers
  adapter-only rules such as python refusing a `hitl_gate`. The rubric
  says a loop never binds its own body sub-node.
- **Compile-time test verification works on a normal install.** pytest
  was only in the `dev` extra, so `pip install 'rote-cli[dbos]'` had no
  test runner, the generated tests were silently skipped, and a compile
  exited 0 with a failing generated test hidden from the repair loop.
  pytest is now a dependency (`>=9.0,<10`), and a missing runner reports
  the deliverable as unverified in the output and the compile report.
- Repair rounds restate the rubric's write-once rules, so a repair no
  longer rewrites compile-report.md in several one-line Edits.
- A TypeScript impl typed with a closed input interface, which is what the
  implement skill asks for, compiles on every TS runtime. DBOS-TS step
  wrappers took `Record<string, unknown>`, and Cloudflare and Inngest
  passed `unknown` values into the typed parameter, so following the
  skill failed tsc. Each value is now asserted to the type the impl
  declares for its key, and the DBOS-TS wrapper takes the impl's input
  type, so a key the impl declares that the IR never sends, or the
  reverse, still fails tsc.
- **Fixes from an end-to-end newcomer pass over main.**
  - Stub detection counts a function whose last top-level statement is an
    unconditional `raise NotImplementedError` and that never returns a
    value, preamble allowed. BDR's `zoominfo.enrich_batch` validates its
    batch size first and was missed, so `rote emit` reported three stub
    modules instead of four and `rote run` would not refuse. A raise on
    one branch, or a dispatch that returns on its real paths, is still
    working code.
  - A missing required input is one line on python and dbos, as on
    temporal: the app checks the input where it reads argv, prints
    `error: pipeline input is missing required field(s): ...` and exits
    2 (dbos before it launches), and `rote run` reports just that line
    instead of a traceback (about 5 KB on dbos).
  - `rote emit`'s `next:` hint is a command that runs: it adds `--input`
    with the `input.json` beside the pipeline (or one level up, the
    examples' layout) when the pipeline has required inputs, and a
    placeholder naming the fields when there is no file.
  - `rote example` also copies `scorecard.md` and `runs/`, and rewrites
    the copy's README so its commands and layout name the copy's own
    paths, not `examples/...` and `/tmp/...`.
  - Missing-extra hints follow the install: `uv tool install
    'rote-cli[<extras>]'` (keeping the extras the tool already has) for a
    uv tool install, `uvx --from 'rote-cli[<extra>]'` under uvx, pip
    otherwise.
  - The stub refusal's fix is worded per language: TypeScript apps are
    told to implement the TypeScript files, not to put Python modules
    beside the pipeline.
  - `rote doctor` labels "api driver" and the "[api] extra" apart.
  - Temporal runs use the app dir as their working directory, so
    expense-audit's report lands in the app like on python and dbos, and
    the dev server's own startup WARN ("Cluster Id ... not a valid uuid")
    is no longer printed.
  - The scorecard's sampled-steps row names its units (agent steps
    before, pipeline nodes after) and says why the totals differ.
  - An invalid `\/` escape in the Node MCP helper template no longer
    warns on import (the emitted TypeScript is byte-identical).
  - The wheel-contents test builds with uv when available, then
    `python -m build`, then pip, so it runs in a uv-made venv.
- **An agent loop now produces its declared output keys, on every
  runtime.** Downstream nodes read a loop's declared `output` keys
  (`process_invoices_loop.output.invoice_results`), but the loop returned
  `{result, provider, iterations}`. The next step raised KeyError on the
  Python runtimes, got `undefined` on Cloudflare (a run reported
  `"undefined / NaN"`), and DBOS-TS and Inngest failed tsc with TS7053.
  The loop now hands its output back through a `submit_result` tool whose
  schema is the declared output (the CLI lane uses `--json-schema`), and
  the step returns exactly those keys, failing with the node and the
  missing keys when the agent does not deliver them. A single-type output
  is returned bare, and a loop with no `output` still returns
  `{result: <text>}`. Provider and turn count move to the usage log
  (Python, `iterations`) and stderr (TS). The emitted TS loop functions
  are typed by their declared keys. Python agent loops also honor
  `ROTE_BASE_URL_<ID>` now, as the TS loops and every judge already did.
  Every lane validates the submission against the declared schema
  (types, required keys, no extra keys) and fails with one message, the
  submit tool is a strict tool when the schema allows, and a loop that
  ends without submitting gets one more turn with `tool_choice` forced to
  it. On the TS runtimes, a type the IR names but never defines
  (`InvoiceResult`) is an alias in a new user-editable `src/types.ts`,
  `JsonValue` until narrowed, and `rote emit` notes each one. The
  `anthropic` floor is now 1.15 (the `api` and `dev` extras, and the
  python runtime's requirements.txt), and the lock moves from 0.116.0 to
  1.15.0.
- A `claude-cli` judge or agent loop now gets a dollar figure. Its usage
  rows were logged under the requested model, often an alias (`sonnet`)
  that no price catalog lists, so `rote run` and `rote eval --run` could
  never price the subscription lane. The emitted inference helper now
  logs one row per model from the CLI envelope's `modelUsage`, under the
  concrete id (`claude-sonnet-5-5`), with the CLI's own `cost_usd`.
  Pricing takes that estimate as is, since it bills a 1h cache write at
  twice the 5m rate a catalog publishes (a measured call: $0.0060 from
  the CLI against $0.0040 from catalog rates). A row with no `modelUsage`
  keeps the requested id, and an alias is still left unpriced rather than
  guessed. `rote run`'s judge-usage line now ends with the dollar figure,
  or says why there is none.
- **`rote mcp login` holds its callback port from selection to callback.**
  The port was checked for being free, then bound later by fastmcp's
  callback server, after the browser had already been sent to it. Anything
  that took the port in between (seen in CI: a concurrent login) killed
  the login with EADDRINUSE instead of it using the next published port.
  The login now binds the port while selecting it (33418 to 33421, both
  IPv4 and IPv6 localhost), runs the callback server on those sockets, and
  releases them on every path, so sequential logins never leak a port.
- `rote run` on an Inngest app reports the error that failed the run
  (for example `RoteMcpConfigError: no endpoint for MCP server 'vendor':
  set ROTE_MCP_VENDOR_URL ...`), read from the dev server's run record
  with its cause chain, instead of the last 500 characters of the app
  log, which were a stack trace with the message cut off above them.
- The MCP URL resolution order is documented as the code runs it:
  `ROTE_MCP_<SERVER>_URL`, then the registry, then the IR `url`
  (Workers: the variable, then the IR `url`). The registry module, the
  IR field descriptions and docs/mcp-client.md said the IR `url` came
  first. No behavior changed for runs. `rote.mcp.resolve_server_url`,
  which encoded that wrong order and had no caller outside its own
  test, is removed.
- An emitted DBOS-TS app whose `DBOS.launch()` fails now exits with the
  error. It used to hang: the half-launched SDK kept its pools and timers
  alive and no shutdown ran. `rote run`'s throwaway Postgres for dbos-ts
  also checks readiness over TCP now. The socket answered during the
  image's init phase, just before the server restarts, which is how CI
  reached the hang. A runner timeout now carries the app's output too.
- Python no longer reads `NaN`, `Infinity` or `-Infinity` in an MCP text
  result as a float. They are not JSON, the TS helpers' `JSON.parse`
  rejects them, and a non-finite float would not survive a JSON step
  boundary, so both runtimes now keep the raw text. The cross-runtime
  payload test also covers empty content, an empty text block, an error
  result, structured content next to text, and an embedded resource, and
  runs the Workers helper itself. The TS integer-precision limit (past
  2^53) is documented in `docs/mcp-client.md`.
- **MCP auth hardening.**
  - The client metadata document check has a 5 second total deadline.
    httpx's timeout is per read, so a host sending a byte every few
    seconds held the login open indefinitely.
  - `rote mcp add --client-id` records an issuer only from metadata that
    describes this resource (RFC 9728 section 3.3) and only over HTTPS
    (loopback excepted), for both the resource and the issuer. An
    origin-level document on a shared host, or a plain-HTTP answer a
    network attacker could write, no longer chooses which server the
    client secret is offered to. `--issuer` must be HTTPS too.
  - The TS refresh grant and `rote mcp export` take the client id and
    secret from the same source, so a stored dynamically registered
    client is never paired with the registry's pre-registered secret.
  - A stored login URL that does not parse (a bad port) parks the run
    on both Python and TS instead of crashing it.
  - When rote sets aside a client from an older token file and the
    server no longer allows registration, the login error says so and
    names the `rote mcp add --client-id` fix.
- python and dbos apps survive a fork after their first async step. The
  background event loop added in this release lives on a thread, and a
  forked child (gunicorn or uvicorn pre-fork workers serving an emitted
  app) inherited the loop but not the thread, so its first async call
  blocked forever. The child now resets the loop state through
  `os.register_at_fork`, where the platform has it.
- The TS MCP helpers (Node, Workers and the rote cloud binding) read a
  text-only tool result by the Python helper's rules. They used to read
  only the first text block, throw on text that wasn't JSON, and return
  `{}` when there was no text, so one MCP step returned different values
  on the TS and Python runtimes. Now every text block is parsed, non-JSON
  text is kept verbatim, several blocks give a list, and no text gives
  `null`, from one TS source spliced into all three helpers.
- Running the test suite no longer rewrites committed files. Five
  adapter suites emitted the BDR pipeline straight into
  `examples/bdr-outreach/expected/runtimes/`, and the dbos and temporal
  ones replaced the example's real `extracted/` implementations with
  stubs, so every `pytest` left a dirty tree. They emit into temp dirs
  now, `tests/test_committed_snapshots.py` diffs a fresh emission
  against each snapshot, and the suite fails if anything under
  `examples/` changes during a run.
- An optional pipeline input can be omitted on every runtime. The
  Python runtimes read input fields as `pipeline_input["f"]`, so
  ops-report's optional `report_date` raised `KeyError` when a caller
  left it out. An omitted optional field is now left out of the step's
  payload, so the impl's default (or a judge schema's default) applies,
  as on the TS runtimes. Every runtime now checks the input on entry and
  fails with `pipeline input is missing required field(s): ...`, or
  `pipeline input must be a JSON object` for a run with no payload, where
  TS used to pass `undefined` through. Temporal raises a non-retryable
  `ApplicationError` and Inngest a `NonRetriableError`, so a bad input
  fails once instead of retrying. New derived IR properties
  `Pipeline.required_input_fields` / `optional_input_fields` define the
  split, and a field listed as both required and optional is rejected.
  A long Node failure excerpt (DBOS-TS) keeps the `Error: ...` line that
  a Node stack prints above its frames.
- An agent loop's local tools now return their result to the model as
  JSON text. The SDK tool runner iterated a dict result, so the model saw
  only its keys.
- Stub detection no longer refuses a filled-in module that only mentions
  the `[rote stub]` marker in a comment or docstring: the marker counts
  in Python only as a `raise` argument, and in TypeScript only outside
  comments. Agent-written stubs that are class methods (a body of only
  `raise NotImplementedError`) are now detected; `@abstractmethod`s are
  not stubs.
- **Token refresh reaches authorization servers off the MCP server's
  origin.** Before discovery the SDK refreshes at
  `<resource-origin>/token`, which 404s for a path issuer or another host,
  so the provider fell into a full authorization and an emitted workflow
  parked on a credential that was fine. The CLI and the emitted Python
  helper now send the refresh to the stored `token_endpoint`.
- **A stored login is sent only to the endpoint it was made for.** Token
  files are keyed by server name, so after `rote mcp add <name>
  <new-url>` or a `ROTE_MCP_<SERVER>_URL` override the old endpoint's
  bearer token went to the new host. A file for another endpoint is now
  treated as absent: the CLI refuses with a `rote mcp login` hint, and
  the emitted Python and Node helpers park the run.
- **Stored clients are bound to their authorization server.** The SDK
  binds only through `client_info.issuer`, so a client stored without one
  matched every issuer and could be offered to a repointed server. The
  stores now fill it from the file's `issuer`, login writes both, and an
  interactive login re-registers a client that has no issuer at all.
- The emitted Python helper never registers a client from workflow code;
  a server that wants a new registration parks the run like any other
  dead credential.
- **Park-on-auth works again under fastmcp 4.** A 401 now reaches the
  caller as a status-free `MCPError`, so the emitted MCP helper's
  `httpx.HTTPStatusError` check never matched and an expired credential
  failed the run instead of parking it. The helper now reads the status
  from an httpx2 response hook on its own transport.
- **MCP step results are plain JSON.** The emitted helper returned
  fastmcp's hydrated `result.data`, which for a typed output schema is a
  generated class that DBOS cannot pickle and that renames non-identifier
  keys. It now returns the wire `structured_content` (unwrapping
  fastmcp's `{"result": ...}` wrapper only when marked), else the parsed
  text content.
- **TypeScript MCP steps read fastmcp results the way Python does.** The
  Node and Workers helpers returned `structuredContent` as-is, so a
  list-returning fastmcp tool was a list in Python and `{result: [...]}`
  in TypeScript. Both now remove the marked wrapper through one spliced
  function; a cross-runtime test runs both helpers against one server.
- A cancellation or interrupt that arrives after a 401 now cancels the
  MCP step instead of parking it as an auth failure.
- `rote mcp login` and token refresh no longer end on `ping`, which the
  sessionless 2026-07-28 protocol does not route.
- `rote serve` keeps live tool-list updates on fastmcp 4: handshake-era
  clients still get `notifications/tools/list_changed`, and sessionless
  2026-07-28 clients get it on a `subscriptions/listen` stream (the
  server now advertises `tools.listChanged` on that era).
- `--runtime python` and `--runtime dbos` now run `async def` extracted
  functions. Steps called every implementation directly, so an async one
  (BDR's hand-written clients are async) returned an unawaited coroutine
  and the run died downstream with `'coroutine' object is not
  subscriptable`. Steps now go through an emitted `_call_impl`, and all
  async work in those runtimes (async impls, dbos MCP steps, legacy
  async judges) runs on one background event loop per process instead
  of `asyncio.run` per call. So a module-level async client (an
  `httpx.AsyncClient`) keeps working across fan_out elements and
  retries, and emitted steps work when called from inside a running
  loop. Temporal shares the same helper's async form, which now also
  awaits a coroutine returned by a plain function.
- `rote emit` (and `analyze`/`compile`) no longer warns "not
  authenticated, the run will park" for an MCP server that needs no
  auth. rote now learns each server's requirement and keeps it in
  `mcp-auth-learned.json` beside the registry (0600, atomic writes),
  keyed by server name and normalized URL so repointing a name
  invalidates it. `rote mcp login` records it: a server whose
  `tools/list` succeeds with no challenge (the OAuth provider discovered
  nothing) is `none`. Login used to fail there with "completed without
  storing a token", and still does when a challenge happened but no
  token landed. `rote mcp add --auth oauth|none` sets it explicitly. A
  server learned as `none` reports "no auth required" (dated in
  `mcp list` and `doctor`) with no recommendation, and live compiler MCP
  tools include it. A tokenless server with nothing learned reports
  "auth unknown", worded as "if it needs auth, the run will park". Only
  a server known to need OAuth gets "will park". Evidence overrides a
  learned `none`: a token for the endpoint wins, and an observed auth
  park records `oauth`. A token file left by the endpoint a name used to
  point at no longer counts as a login, in `login` or in the status. The
  advisory's `"unknown"` state for an unreadable registry is now
  `"registry unreadable"`.
- **Version compatibility of `~/.config/rote/mcp.json`.** Released rote
  versions up to 0.14.0 load the registry with unknown fields rejected,
  so a field a newer rote writes breaks every older rote on the same
  machine (a cached `uvx --from rote-cli`, the plugin's `uv run`). That
  is why learned auth lives in its own file and the registry schema is
  unchanged by it. `rote mcp add --client-id ... --issuer ...` (new in
  this release) does write an `issuer` field, so a registry with an
  issuer-bound client is unreadable to 0.14.0 and older until that
  entry is removed. From this release on, the registry and the sidecar
  keep fields they do not know and read unknown `transport`/`auth`
  values as unsupported or unknown instead of rejecting them, and every
  writer keeps a higher `version` than its own.
- README and AGENTS.md accuracy: the compiler skill has six reference
  files, not four; `rote run` executes all six runtimes (AGENTS.md said
  python/dbos only); the slow e2e suites do run in CI; the BDR run flags
  three mandatory nodes, not four.
- A `fan_out` node can fan over a pipeline-input list
  (`invoice: pipeline.input.invoices`). The element resolver only knew
  upstream node outputs, so emission raised, and `rote compile` crashed
  with a traceback after the agent had been paid.
- No failure after a paid agent run loses the IR. An emission error
  prints the cause and the `rote emit` command that retries it, and an
  IR the validator rejects is kept in the output dir (an update's goes to
  `<out>.rejected`, leaving the previous output alone) instead of being
  deleted with the temp work dir.
- Emitted Python agent loops report the turns that actually ran. The
  vendor-SDK lane returned `max_iterations` as `iterations`, so every loop
  looked saturated, and it logged only the final request's tokens to
  `$ROTE_USAGE_LOG`. It now counts the assistant messages the tool runner
  produced (the TS twin's semantics) and sums usage over every turn. The
  `claude -p` lane reports `None` instead of `0` when the envelope carries
  no `num_turns`.
- `uvx rote-cli ...` works. The wheel installed only a `rote` command,
  and uvx runs the command named after the package, so `uvx rote-cli
  serve` failed with executable not found. The package now installs
  `rote-cli` as a second command for the same CLI.
- `--runtime temporal` now emits an `extracted/` package, the same way
  dbos and python do: the agent's real modules from beside
  `pipeline.yaml` where they exist, IR-derived stubs otherwise. Before,
  activities imported `expected.extracted.*`, which resolves only inside
  the BDR example. So `rote emit --json` reported no
  `unimplemented_stubs`, `rote run` could not import a single module,
  and an agent loop with tools lacked `extracted/_rote_mcp.py`. Plain
  (non-async) implementations now run in a worker thread instead of
  being awaited, which used to raise `TypeError`.
- **A judge whose `inputs:` cannot satisfy its input schema is rejected
  at load time.** Every adapter builds a judge's payload from its
  `inputs:` keys alone, so a `required` property nothing binds failed
  validation on every call, and nothing caught it before run time. A
  bound param the schema does not declare is rejected too, whatever
  `additionalProperties` says: a binding the judge's contract cannot
  name is a mistake either way. Found by a real eval: deal-monitor's fanned
  `classify_thread_step` bound only `thread` while its schema also
  required `account_name`, so all of its calls failed. The new
  `Pipeline` validator names the node, the params, and the schema, and
  on a `fan_out` node names the element param through the same
  `fan_out_element_param` the emitters use (moved to `rote.ir` so the
  IR and the adapters share one resolver). Loop-body sub-nodes and
  legacy `signature:`-only judges are exempt. Four committed IRs failed
  it and are fixed: deal-monitor `expected` (the judge now takes the
  whole `MatchedThread`, which carries the account), deal-monitor runs
  `2026-07-18-rubric-v2` and `2026-07-18-first-agent-run` (fanned
  judges had their element flattened into top-level fields), and BDR
  run `2026-04-26-signature-spec` (`personalize_email` bound nothing).
  The compiler rubric now says a fanned judge's input has one property
  for the whole element, and shows the wrong shapes it produced.
- **Zod schemas honor `additionalProperties` the way Pydantic does.**
  The TypeScript emitters put `.strict()` on every object, while the
  Python emitters forbid unknown keys only under
  `additionalProperties: false`. An upstream object carrying one
  undeclared field therefore passed on dbos and python and failed every
  judge call on Cloudflare, Inngest and DBOS-TS. Objects now get
  `.strict()` only when the schema is closed and Zod's default strip
  otherwise (Pydantic's `extra="ignore"`), and a property-less object is
  `z.record(z.string(), z.unknown())` to match `dict[str, Any]`. A new
  slow test runs one schema through the emitted Pydantic model and real
  `zod` under Node and requires the same verdicts and parsed values.
- deal-monitor's `expected` IR no longer sets `temperature: 0.0` on its
  three judges. Current Claude models reject the parameter, and the
  rubric already forbids it. A test now keeps every `expected/` IR free
  of it; historical `runs/` snapshots stay as compiled.
- **The subscription lane no longer carries the user's own Claude Code
  setup.** Emitted judges and agent loops on `claude-cli` still loaded the
  user's claude.ai connectors and plugin MCP servers, and the auto-memory
  index of the repository they ran from, despite `--tools ""
  --setting-sources ""`. Measured on Claude Code 2.1.302, a one-line judge
  was 30.3k tokens and $0.183 a cold call; with `--strict-mcp-config` and
  auto memory off it is 1.25k. Agent loops take the same flags plus their
  own `--mcp-config`, and now also `--tools ""` (it removes only built-in
  tools), so a loop sees its declared servers and nothing else: a
  two-turn loop went from 120k tokens / $0.38 to 2.8k / $0.012 with its
  MCP tool still bound. Both paths share one flag tuple.
- **`rote compile` with the claude driver passes `--strict-mcp-config`.**
  The driver wires no MCP, yet the user's connectors put 357 tools in the
  first request, past 200k tokens, and a subscription without usage
  credits rejected the compile before it started. Strict, the request is
  70.7k tokens.
- **`$ROTE_USAGE_LOG` records all four token buckets**
  (`input_tokens`, `output_tokens`, `cache_write_tokens`,
  `cache_read_tokens`), and `rote eval --run` prices all four at live
  rates. A prompt-cached `claude -p` judge used to log 3 input tokens.
  A cache bucket with no published rate marks the model unpriced instead
  of guessing. `rote run`'s judge-usage line counts cache tokens as input.
- `rote eval` prices a `fan_out` node per element. The estimator gave
  every non-loop node one call per run, so a fanned judge (BDR's
  `personalize_email`, deal-monitor's `classify_thread_step`)
  understated the after-side dollars by the fan width. The width comes
  from the compiler's `eval.yaml` (`iterations` on the node's step)
  when it records one, else from the new `fan_out_elements_per_run`
  prior (default 10, `--prior` to override). The scorecard's
  Assumptions section lists each fanned node's width and source. Elements
  run concurrently, so they multiply tokens and dollars but not wall time.
- **First-run CLI output, from a fresh-install test of 0.13.0.**
  - Every install hint now says `rote-cli[...]` (`rote doctor`, the api and
    openai-api drivers, `rote serve`, the serve backends). The `rote`
    package on PyPI is an unrelated library. All hints come from
    `rote.extras.install_hint`; a test fails on a bare `rote[`, or a
    hand-typed `rote-cli[`, anywhere in `src/` outside `rote/extras.py`.
  - `rote run` checks the runtime's extra before starting (dbos, temporal)
    and prints the install hint instead of a `ModuleNotFoundError`
    traceback. When an app subprocess dies on a missing module that an
    extra provides, the error ends with the same hint.
  - `rote run` refuses an emitted dir whose `extracted/` modules are still
    stubs, listing the files, before any retries run. Previously a stub's
    `NotImplementedError` was retried for about 10 seconds first.
    `--allow-stubs` starts the run anyway. A stub is a function that raises
    with the `[rote stub]` sentinel rote now puts in every emitted stub, or
    (Python) one whose whole body is `raise NotImplementedError`, so a real
    module that raises it on an unsupported branch is not refused.
  - A failed runner subprocess now shows the end of its output, where the
    exception is, not the first 800 characters of startup logs cut off
    mid-word.
  - Plain-text `rote emit` and `rote compile` print the stub count, each
    stub path, and the next `rote run` step. Text, `--json`, and the
    preflight read the same manifest-based stub list. Output whose code
    imports its implementations from outside the emitted dir says the stubs
    are "not tracked" (and `--json` carries `stubs_tracked: false`) instead
    of reporting none.
  - `--json` commands that fail without printing a result now write
    `{"ok": false, "exit_code": N, "error": "..."}` to stdout, keeping the
    exit code. This is enforced once in `main()` for every `--json` command.
  - `rote run --help` lists the runtimes it actually runs (all six), and
    `--model` help reads the default model from the driver constant.
  - Emitted READMEs (dbos, dbos-ts, python) document the env knobs of the
    pipeline's real judges, or say there are none, instead of a
    `ROTE_MODEL_VET_CONTACT` example on pipelines with no judges.
  - `rote config` columns line up for every key, including `inference`.
- A missing MCP endpoint (no `ROTE_MCP_<SERVER>_URL`, no registry entry,
  no URL in the IR) now fails the step at once on every MCP runtime,
  naming the variable to set. It used to be retried like a network
  error: DBOS ran all 4 attempts and reported only
  `DBOSMaxStepRetriesExceeded`. The helpers raise a new
  `RoteMcpConfigError`, which `should_retry` (DBOS Python),
  `shouldRetry` (DBOS-TS), `NonRetriableError` (Inngest) and
  `NonRetryableError` (Cloudflare) now exclude from the retry budget,
  the same way they already excluded auth failures. It does not park:
  the fix is the variable and a restart, not a login. Agent loops get
  the same treatment: tool discovery that cannot reach a pinned server
  for lack of an endpoint raises `RoteMcpConfigError` (TS and Python),
  and each runtime's loop step skips its retries on it (temporal via
  `non_retryable_error_types`). The Python loop also wires a pinned
  server that resolves from its `ROTE_MCP_<SERVER>_URL` alone, which
  it used to ignore.
- `rote run` on a Cloudflare app reports the step error that failed the
  run. Local dev's status only says `WorkflowFatalError`, so the runner
  now reads the real error back from the wrangler log. A step sleeping
  between retry attempts after the last gate payload was delivered is
  no longer mistaken for a park and failed with "every gate payload has
  been delivered"; the runner waits for the run's real outcome, reports
  the newest step error, and still fails fast on a gate that never
  consumes its payload while nothing is retrying.

## [0.14.0] - 2026-10-18

### Added
- **`rote run` shows an MCP auth park instead of hanging.** On dbos,
  dbos-ts, inngest and cloudflare, a run that parks on a missing or dead
  MCP credential prints which node is waiting, on which server, and the
  command that fixes it. On a terminal it offers the browser login in
  place and releases the run. Piped or in CI it never prompts and
  instead releases the run once the server's credentials change (for
  example after `rote mcp login` in another terminal). Emitted parks now
  name their node: the DBOS `rote_auth_status` event carries `node` and
  a per-run `park` number, and
  the Inngest and Cloudflare parks print a `rote-auth-park` marker line
  before waiting.
- `rote mcp parked [server] [--json]` lists workflows parked on auth
  across the registered DBOS apps without releasing them.
- `rote compile --no-mcp` skips live MCP discovery and calls for local
  `api` and `openai-api` compilation. `rote analyze` accepts the same
  opt-out. Emitted MCP bindings and default discovery behavior stay
  unchanged; cloud compilation, subprocess drivers, and `--baseline`
  reject the flag with an actionable error.
- **Invalid agent-authored pipelines are repaired, not failed one-shot.**
  When the driver's produced pipeline.yaml fails IR validation, the
  Compiler resumes the same agent conversation with one user turn
  carrying the verbatim pydantic errors and a change-only-what-the-
  errors-name instruction, then re-validates — up to two repair attempts
  (each surfaced as a warning event) before failing with the last error
  exactly as before. Drivers that can resume their conversation take an
  optional `repair` callback (api and openai-api do); subprocess drivers
  keep the one-shot behavior. Seen twice in real compiles: `.ts` module
  refs in `impl:` and dict-typed node input/output entries — both
  trivially fixable by the model that wrote them.
- **Heartbeat events while awaiting a model request.** The api and
  openai-api drivers emit a `log` event every 120 seconds while an LLM
  request is pending, so a hosted hang detector can tell a slow or
  retrying request from a wedged job.
- **The api and openai-api compiler drivers can call MCP tools live.**
  A new driver kwarg `mcp_servers` (each entry `{"name", "url",
  "headers"}`, streamable HTTP only) lists each server's tools at agent
  start and exposes them to the compiler agent as `mcp__<server>__<tool>` —
  gated to tools whose server declares `readOnlyHint`, the same
  predicate `rote baseline` already applies, now shared in
  `rote.mcp.live_tools` so the two gates cannot drift. Every tool call
  opens a fresh connection — long-lived streamable-HTTP sessions
  combined with the vendor SDK deterministically froze the asyncio loop
  under the hosted container runtime. Credentials are resolved once
  before compilation and reused for every call. A server that cannot be
  reached is a warning event, never a failed compile, and a tool failure
  returns to the model as an error tool result. When `rote compile` runs one of
  these drivers and no servers were passed, the local registry's
  authenticated servers are wired in automatically (static headers
  verbatim, logged-in servers via a freshly refreshed token;
  unauthenticated servers are skipped with a printed reason).
- **A wedged MCP server can no longer freeze a compile.** The 180s bound
  wrapped only `call_tool`, leaving the initialize handshake unbounded,
  and since every call opens a fresh connection that handshake runs on
  every call. Startup tool-listing had no bound at all. A server that
  accepts the socket and never answers raises nothing, so the
  "unavailable, continuing without it" path could not fire and the
  compile hung with no warning. Both connects are now inside the bound.

- **Binding-backed Workers MCP helper.**
  `get_adapter("cloudflare").emit(pipeline, out_dir, mcp_client="binding")`
  (also available as a factory option, `get_adapter("cloudflare",
  mcp_client="binding")`) emits a `_roteMcp.ts` variant with the same
  exported surface as the direct helper, implemented over a
  platform-provisioned `ROTE_MCP` RPC service binding: no per-server
  secrets, no `ROTE_MCP_TOKENS` KV namespace, no OAuth refresh, and no
  MCP SDK dependency. Every proxy call carries a signed caller-auth
  object (`{tenant_id, pipeline, run_id?, sig}`) built from the
  dispatcher-injected `ROTE_TENANT_ID` / `ROTE_PIPELINE` / `ROTE_RUN_ID`
  / `ROTE_MCP_SIG` variables, so tenant isolates cannot impersonate each
  other; a missing `ROTE_MCP_SIG` is a config error naming it. The
  generated Env interface declares the one `ROTE_MCP` binding plus those
  four variables instead of the per-server secret surface. An error
  whose message starts with `ROTE_MCP_AUTH_NEEDED:` — the platform
  proxy's contract — is rethrown as `RoteMcpAuthNeeded`, so the
  park-on-auth loop, the `rote_auth_<server>` event names, and agent
  loops are unchanged. The default (`"direct"`) emission is untouched.
- **`manifest.json` carries the pipeline's required MCP servers.** A new
  `mcp_servers` key (server name → sorted bound node ids, from
  `Pipeline.required_mcp_servers`) is always present in the Cloudflare
  manifest — `{}` when the pipeline makes no MCP calls — so the cloud
  runner can provision connections without re-deriving them from the IR.

### Fixed
- The `dbos` and `dev` extras now cap `dbos<3`. DBOS 3.0 (2026-09-16)
  made `Queue(...)` construction raise in favor of
  `DBOS.register_queue` and removed the admin server, so a fresh
  `pip install 'rote-cli[dbos]'` resolved 3.x and every emitted DBOS
  app died on import. The lock moves to 2.31.1, the newest 2.x. This is
  a stopgap until the emitter targets DBOS 3.
- **Emitted TypeScript compiles once you fill in a stub.** The Cloudflare,
  DBOS-TS and Inngest workflows cast every exit result with
  `as Record<string, unknown>`, so a node declaring a scalar output (for
  example ops-report's `generate_report: report_markdown str`) failed
  `tsc` with TS2352 as soon as its implementation returned
  `Promise<string>`. Field access (`foo.output.f`) had the same hole for
  an `interface` result, and a `fan_out` exit failed even against the
  stubs. Exit results are now returned uncast and field access indexes
  the result directly, from shared helpers. With no cast, tsc also checks
  each IR field reference against the implementation's declared type, so
  a renamed field is a compile error instead of `undefined` at run time.
  It compiled before only because stubs return `Promise<never>`, and
  `rote run` passed because casts are erased at runtime. A new slow suite
  (`tests/test_ts_typed_impls_e2e.py`) replaces the stubs with typed
  implementations of every output shape and runs `tsc` on all three
  runtimes, plus a drift case that must fail.
- Python runtime emission keeps input and output signature definitions in
  separate namespaces. Schemas may reuse a `$defs` name with different
  definitions; generated model names remain unique without changing the
  output schema sent to the model. This fixes emission of the preserved
  core-skill compilation on Python, DBOS, and Temporal.
- Final compiler metrics now survive runtime-emission failures in
  `compiled/compile-metrics.json`. The atomic sidecar contains allowlisted
  numeric/identifier fields, preserves incomplete usage, labels cost as an
  estimate, and excludes prompts, credentials, and driver error prose.
- Claude compiler progress counts each response ID once, retaining every
  tool event without multiplying cached input across content blocks.
  Live output reads "pending" and live USD is omitted until final usage
  arrives. Final token totals use the CLI's modelUsage/usage fields;
  interrupted streams retain input/cache totals with output null and
  usage_complete false. The CLI's total_cost_usd remains a cost estimate,
  not a billing record.
- **CLI cloud uploads preserve required MCP servers.** `rote deploy`
  now forwards the emitted manifest's server map, so the platform can
  authorize the pipeline's MCP calls. Binding-mode run IDs are optional
  attribution data; generated types and comments match the proxy's
  tenant-and-pipeline signature contract.
- **Cloud uploads can bundle packages with a legacy entry point.**
  esbuild's neutral platform has no default package entry fields.
  Explicit `module,main` resolution lets generated agent loops bundle
  `@cloudflare/ai-utils` while preserving Worker export conditions.
- **The readOnlyHint gate no longer trips fastmcp 4's deprecation
  warning.** MCP SDK v2 renamed `ToolAnnotations.readOnlyHint` to
  `read_only_hint`, and reading the camelCase alias fires
  FastMCPDeprecationWarning (it surfaced as stderr noise in prod error
  mail). The shared predicate now reads snake_case first and falls back
  to camelCase, clean under both fastmcp 3.4 and 4.x.
- **Long compiler turns no longer die on silent connections.** The api
  and openai-api drivers stream every model request (`messages.stream()`
  / `stream=True` with usage included) and assemble the final message
  into the exact shape the loop already handled. A multi-minute
  non-streaming generation sends no bytes until it finishes, so
  intermediaries timed the connection out and the SDK's retries hit the
  same wall — a real compile's pipeline-authoring turn sat 720 seconds
  and failed with APIConnectionError; the 32k thinking-friendly turn cap
  is what made single turns long enough to die. The client timeout now
  bounds the gap between streamed chunks (httpx read-timeout semantics),
  not the total generation, and the 120s heartbeat fires only when a
  stream actually stalls.
- **Pinned `fastmcp` below 4.** The declaration was `fastmcp>=3.4.2` with
  no upper bound, so CI resolved to `fastmcp` 4.0.2 (published
  2026-09-02) and `mcp` 2.x on every fresh install while local
  environments stayed on 3.x. Five tests across three CI jobs began
  failing on an unchanged `main`: the `serve` client stops receiving
  tool-list-changed notifications, two OAuth e2e tests fail, and the TS
  runtime e2e reports `Method not found`. The pin restores a trustworthy
  signal; migrating to the 4.x / 2.x line is separate work, and MCP SDK
  v2 also renames `Tool.inputSchema` to `Tool.input_schema`.
- **A compilation's reported tokens and cost ignored prompt caching.**
  `claude -p` always runs prompt-cached, so `input_tokens` holds only the
  handful of tokens that were neither written to nor read from the cache.
  The driver summed that field alone, and the CLI priced the result: a
  22-minute, 62-turn compilation was reported as `tokens in=113 out=1919;
  cost $0.029`. All four token buckets are now counted, and every display
  adds cache writes and reads into the input figure it shows.
- **The agent runtime's own estimated cost was discarded.** The driver read
  `result["cost_usd"]`, a key the `claude -p` result envelope does not
  have. The field is `total_cost_usd`, so the read returned `None` on
  every run while the estimate sat one key away. It is now read and
  preferred to rote's single-model estimate because it includes the
  per-model usage split. Neither estimate is a billing record.
- **Cache rates are carried for every fetched model, not just the three
  tier representatives.** models.dev publishes them per model and the
  parser already read them; `build_catalog` then discarded all but the
  representatives'. The compiler's own default model is not one, so a
  default compilation had no priceable cache rate at all. New
  `PricingCatalog.rates_for` spans the whole fetch; an on-disk cache
  written before this still loads, degrading to "no published rate".
- **A run that still cannot be priced honestly reports no cost.** When a
  run touched the cache and its cache rates are genuinely unknown, the
  cost field is omitted rather than filled with a figure that would
  overstate cache reads roughly tenfold.

### Changed
- The README roadmap dropped two items that had already shipped: real
  compiler runs have produced `signature_spec` since April, and the
  ops-report, deal-monitor, and invoice-push archetypes have existed
  alongside BDR for some time.


## [0.13.0] - 2026-07-31

### Added
- **`fan_out` dispatches per element on all six runtimes.** A node marked
  `fan_out: true` is invoked once per element of its bound list, and the
  result binds as a list in input order. This worked on DBOS only; the
  other five handed the whole upstream list to a single invocation, which
  meant a judge signature had to accept the batch and the "one call per
  element" the IR describes was a fiction everywhere else. Per runtime:
  DBOS enqueues one durable step per element, python uses `pool.map`,
  Temporal `asyncio.gather` over one activity execution each, Cloudflare
  and Inngest `Promise.all` over per-element steps, DBOS-TS
  `Promise.allSettled` + unwrap.

  Which input is the fanned list is resolved by
  `rote.adapters._common.fan_out_element_param` — fan_out edge marker >
  incoming-edge source > only node-bound param, with ambiguity an
  emit-time error rather than a guess. It lives in the language-neutral
  common module on purpose: two adapters disagreeing about which input is
  the list would make one `pipeline.yaml` mean two different things.

  Two traps worth knowing if you read the emitted code. Cloudflare and
  Inngest key a durable step by its *name*, so each element emits
  `` `<id>[${_index}]` ``; reusing one name made Cloudflare return element
  0's cached result for all N, so the run succeeded, the output had the
  right length, and every entry was identical. And a parkable (MCP-bound)
  fan_out node on Cloudflare runs its elements sequentially, because
  `waitForEvent` inside a promise combinator is undocumented and its
  timeout throws.
- **An undefined fanned list names the node that expected it.** Emitted
  code routes the list through a guard (`fanOutList` in TypeScript,
  `_fan_out_list` in Python) that throws with the node id and the IR
  reference. Without it the failure was `Cannot read properties of
  undefined (reading 'map')` inside generated code the user never wrote —
  found by running it, on the first live fan_out run. The cause is almost
  always an upstream node that didn't return the key the IR says it does.

### Changed
- **Every slow end-to-end suite now runs in CI.** Three of twenty-seven
  did; Cloudflare, DBOS-TS and Inngest had no automated coverage at all,
  on the reasoning that they need a Node or Docker toolchain that CI
  doesn't have. `ubuntu-latest` ships both, and the premise had never
  been checked. All seven jobs — lint/types/sanity, three Python
  versions, and the Python, TypeScript and DBOS-TS e2e suites — are now
  required status checks.
- **A test can no longer spend inference tokens.** Anything that doesn't
  require inference is a test: automated and fast. Anything that costs
  tokens is an eval: run deliberately, never in CI. That split is now
  enforced rather than trusted — the suite scrubs vendor credentials and
  makes a bare-name lookup of `claude` / `codex` fail. The second half is
  the one that matters, because `claude -p` authenticates from an OAuth
  session, so having no API key was never protection. Two DBOS suites had
  been quietly burning real subscription inference on every run.

### Fixed
- **Fifteen test gaps found by mutation testing**, across three sweeps
  and 64 mutations. These are defects in the suite's ability to detect
  regressions, not in shipped behavior — but each one is a real
  regression that could have shipped green. The most consequential:
  per-node `timeout:` was ignorable on Temporal and Cloudflare gates;
  `DEFAULT_MODEL` could be flipped to Opus (~5x cost) and
  `DEFAULT_MAX_TURNS` halved with nothing failing; the agent-side cost
  estimate could be off by 1000x because it was only ever compared
  against itself; an `agent_loop` could understate its tokens and wall
  time threefold; and `rote compile` could report success while doing
  nothing when handed a bad config.
- `1h` in a per-node `timeout:` rendered as `"1 hours"` in emitted
  comments and docstrings.

## [0.12.1] - 2026-07-30

### Changed
- Metadata-only release. PyPI now carries the rewritten project
  description, the roteskills.com homepage and changelog URLs, the
  richer keyword set, `Development Status :: 4 - Beta`, the
  `Environment :: Console`, AI-topic and `Typing :: Typed`
  classifiers, an author email, and the `py.typed` marker added in
  0.12.0's README and discovery work (#72, #74). No code changes.

## [0.12.0] - 2026-07-30

### Added
- **`agent_loop` is a real node kind on the TypeScript runtimes.** It was
  a stub on all three — `throw new Error("requires an agent runtime —
  implement me")`. A pipeline that hands its one genuinely agentic step
  back to the user has not graduated that step, so Cloudflare, DBOS-TS
  and Inngest now emit the loop: MCP tools bound through whichever MCP
  helper that runtime already emits, `loop_body` sub-nodes bound as
  callables (they are working steps, so the agent drives them per
  iteration exactly as the IR describes), and both handed to
  `runAgentLoop` in a verbatim `signatures/_roteInference.ts`. Three
  inference lanes and deliberately no `claude-cli` one: workerd cannot
  spawn a subprocess, and a lane that worked on two of three TypeScript
  runtimes would be a portability trap.
- **The `workers-ai` lane (Cloudflare only).** An operator can run the
  loop on their own Cloudflare account with no external API key. The `AI`
  binding and its `runWithTools` runner are injected at the call site,
  which keeps `_roteInference.ts` free of any Cloudflare import — the
  Node runtimes compile it without `@cloudflare/ai-utils` on their
  dependency list. Offered, never forced: the binding costs nothing until
  a run selects the lane.
- **`Node.tool_servers` — an agent loop's tools now know their server.**
  `tools` names tools without a server, and `required_mcp_servers`
  derives from server *names*, so a pipeline whose only MCP usage was a
  loop reported **zero** required servers: `analyze`, `emit`, `compile`
  and the `rote mcp login` advisories all said there was nothing to
  authenticate. BDR is exactly that shape, so the flagship example
  demonstrated the bug; it reports five servers now. Additive — partial
  maps are valid and an unmapped tool still resolves at run time, so no
  existing pipeline changes meaning. A resolved tool also binds from its
  own server and no other, because two servers exporting one tool name is
  precisely what an allowlist cannot disambiguate.
- **`rote.probe` resolves those servers from evidence.** `cross_check`
  gained `resolved_agent_tools`: an observed tool that a loop declared
  bare is the *answer* to the gap, not a missed requirement, so it no
  longer reads as "the compiler missed this". `enrich_pipeline` writes the
  server back into `tool_servers`, leaving a tool served by two servers in
  one trial unresolved rather than guessing. An authored value always
  beats inference.

### Fixed
- **The subscription lane was dead on arrival.** Both `claude -p` call
  sites passed a bare `--tools`, but the flag is variadic — Claude Code
  2.1.220 rejects it with `option '--tools <tools...>' argument missing`,
  exit 1, before running anything. That is every judge *and* every agent
  loop on the lane that exists so a laptop run costs nothing extra.
  `--tools ""` still loads no tools (167 input tokens on a one-line
  prompt, so the low-overhead path is intact). Shipped in 0.11.0; no test
  caught it because the assertion checked the flag's presence, not its
  value, and `run_agent_loop`'s CLI path had no coverage at all.
- **Python's `--allowedTools` stopped being a cross product.** It was
  allowlisting every declared tool on every wired server — mostly phantom
  pairs, and a real hazard when two servers export the same tool name.
- **Two DBOS e2e suites were spending real subscription inference to test
  nothing.** Their overlay mocked agent loops by writing an `extracted/`
  module, but `_extracted_layout` deliberately skips agent loops (the
  adapter inlines `run_agent_loop`), so that branch was unreachable and
  BDR's two loops ran for real — about 35s of a 60s budget, orchestrating
  nothing the tests asserted. Both now pass in under 9s.

### Changed
- **Invariant #3 now names both places the IR declares MCP:** a node's
  `mcp:` binding and an `agent_loop`'s `tools:`. The Python runtimes have
  behaved this way since 0.11.0, so this is the documentation catching up.
  The MCP-free scan *names* its legitimate sites rather than skipping any
  file that mentions MCP, so an unexpected reference still fails and an
  expected site that stops referencing MCP fails too.
- **A Cloudflare agent loop is parkable, and therefore sequential.** Its
  tools are MCP tools, so it parks on auth like any bound step; parkable
  steps stay out of `Promise.all` because `waitForEvent` inside a promise
  combinator is undocumented and its timeout *throws*, which would reject
  every sibling. A real throughput cost on the slowest node kind, taken
  so park-on-auth holds on the default runtime.

## [0.11.0] - 2026-07-26

### Added
- **Park-on-auth for Cloudflare Workflows — every MCP-capable runtime
  now parks.** MCP-backed `step.do` calls wrap auth failures in
  `NonRetryableError` (Workflows has no should-retry predicate; a dead
  credential must not burn `retries.limit` worth of delay) and park
  the instance on a `rote_auth_<server>` `waitForEvent` with an
  explicit 30-day timeout — the default is 24 hours and expiry
  *throws*, failing the instance, which after 30 unreleased days is
  the intended outcome. Release has no broadcast on Workflows, but
  events sent before an instance reaches its wait are **buffered
  per-instance**, so `rote mcp release <server>` simply sends the
  event to every non-terminal instance via the Cloudflare REST API
  (`CLOUDFLARE_API_TOKEN` + `CLOUDFLARE_ACCOUNT_ID`; in `wrangler
  dev`, use `npx wrangler workflows instances send-event … --local`).
  The credential fix on Workers is re-provisioning (`rote mcp export`
  + `wrangler secret bulk`), not a local login — the `ROTE_MCP_TOKENS`
  KV cache supersedes secrets, so rotated tokens keep working. Proven
  live on workerd (`tests/test_mcp_park_cf_e2e.py`): instance parks
  with nothing provisioned, ignores a wrong-server event, wakes on the
  real one, and completes — with the retried step driving the real
  `@modelcontextprotocol/sdk` streamable-HTTP client inside workerd
  against a live MCP server, closing the long-standing "typecheck
  only" gap for the Workers MCP output.

- **Park-on-auth for Inngest, released by broadcast.** The Inngest
  adapter emits a `runParkable` loop around MCP-backed steps: auth
  failures are wrapped in `NonRetriableError` inside the step (so
  Inngest's per-step retry budget and managed backoff never burn on a
  missing credential), each attempt gets a fresh memoization-safe step
  id, and the run parks on `step.waitForEvent` for
  `<pipeline>/rote.auth.<server>`. Release needs no discovery: Inngest
  events fan out to every matching waiter, so `rote mcp login` /
  `rote mcp release` send **one** broadcast per registered pipeline
  (dev server by default, Inngest Cloud with `INNGEST_EVENT_KEY`,
  or `ROTE_INNGEST_EVENT_URL`). Because Inngest does not buffer events
  for waits that haven't started, the park loop retries once before
  waiting — a release landing in that gap fixes the credential store
  the retry then reads. Proven live against `inngest-cli dev`
  (`tests/test_mcp_park_inngest_e2e.py`): run parks, a wrong-server
  broadcast leaves it parked, the real broadcast wakes it, and it
  completes with live MCP data.

- **Park-on-auth for DBOS TypeScript, released from Python.** The
  dbos-ts adapter now emits the same park the Python adapter got:
  MCP-backed steps throw `RoteMcpAuthNeeded` (typed error in the
  emitted `_roteMcp.ts` — thrown on dead tokens, failed refresh grants,
  and 401s the refresh can't fix), a `shouldRetry` predicate keeps auth
  failures out of the retry budget, and the workflow parks on
  `DBOS.recv("rote:auth:<server>")` with the wait advertised via a
  *portably-serialized* `rote_auth_status` event — the one format both
  DBOS SDKs read, which is what lets the existing Python release path
  serve TS apps unchanged (it now scans `dbos-ts` registry entries,
  derives the TS Postgres system-DB URL, and sends the release message
  in `WorkflowSerializationFormat.PORTABLE`). New `rote mcp release
  <server>` releases parked workflows without a login, for credentials
  fixed out-of-band. The full cross-language loop — TS parks, Python
  reads the event, Python releases, TS resumes against a live MCP
  server — is proven on a real Docker Postgres in
  `tests/test_mcp_park_ts_e2e.py`. Inngest and Cloudflare parks are
  next.

- **Park-on-auth: workflows suspend on missing MCP credentials instead
  of failing** (DBOS Python). OAuth is interactive; durable workflows
  run unattended — so when an MCP-backed step finds its credential
  missing or dead (expired with no refresh token, a 401 from the
  server, or an OAuth flow demanding authorization — emitted code never
  opens a browser), it raises `RoteMcpAuthNeeded` and the workflow
  parks durably on a `rote:auth:<server>` topic, exempt from the retry
  budget and advertised via the `rote_auth_status` workflow event.
  `rote mcp login <server>` now finishes the loop: after a successful
  dance it scans the new app registry (`~/.local/share/rote/apps.json`,
  recorded by `rote emit`/`rote compile`; `ROTE_APPS_PATH` overrides),
  discovers workflows parked on that server, and releases them — the
  run picks up exactly where it stopped, with the fresh credential.
  Parallel-wave siblings retry once before parking so a stale auth
  failure can't strand a workflow after its release signal was already
  consumed. A workflow parked longer than 30 days times out loudly.
  Proven cross-process on the real DBOS runtime against a live MCP
  server (`tests/test_mcp_park_e2e.py`). The TS runtimes still fail
  loud on dead credentials — extending the park is a known follow-up.

### Changed

- **`rote graduate` is now `rote compile`.** The operation always was
  compilation: read a source artifact, classify its parts, lower the
  deterministic ones into typed code, keep a runtime for what genuinely
  needs judgment. `graduate` described what happened to the skill
  socially rather than what the tool does mechanically, and nobody
  reaching for this searches for it. Asked cold to name the operation,
  both Claude and Codex produce "compile" unprompted; the experiment is
  recorded in the rote-cloud repo under `seo/reports/verb-experiment.json`.

  The rename runs through the whole surface: the command, the
  `rote.compiler` package (was `rote.graduator`), `Compiler` /
  `CompilerError` / `CompilationEvent`, the `skills/rote-compile` agent
  bundle, the `/rote:compile` plugin skill, the `<out>/compiled/`
  artifact directory (was `<out>/graduated/`), the emitted
  `compile-report.md` (was `graduation-report.md`), and all docs.

  **`rote graduate` still works.** It dispatches to `rote compile` and
  prints a one-line notice on stderr. It is deliberately absent from
  `--help`: the alias exists so scripts written against 0.10.x keep
  running, not to offer a second spelling.

  **Breaking, without an alias:** the `--json` payload and progress
  sidecar now carry `cloud.compilation_id` instead of
  `cloud.graduation_id`, and artifacts land under `<out>/compiled/`.
  Anything parsing that output needs a one-word edit.

  **Not renamed:** the rote cloud HTTP routes (`/v1/graduations`) and
  the server response keys (`graduation_id`, `active_graduation`). Those
  name endpoints on a separately deployed platform; renaming them
  client-side would 404 against every released server. See the module
  docstring in `src/rote/cloud_compile.py`.

## [0.9.0] - 2026-07-11

### Added
- **Live compilation progress** — the compiler now emits structured
  `CompilationEvent`s (`rote.compiler.events`): phase transitions
  (driven by `progress.ndjson` markers the compiler skill writes at
  the start of each phase), per-turn token counts, tool calls, and
  artifacts. `rote compile` renders them live on stderr; hosts embed
  the stream via `Compiler(on_event=...)`. Works across the Claude
  subprocess driver (stream-json) and both in-process API drivers.
- **OpenAI-compatible API driver** (`--agent openai-api`) — the same
  in-process compilation loop against any OpenAI-shaped endpoint
  (GPT, GLM, Kimi, …), sharing one filesystem-tool surface with the
  Anthropic driver.
- **Gateway-friendly driver auth** — both API drivers accept
  `base_url` and `default_headers` (through the new
  `Compiler(driver_kwargs=...)`), so compilations can route through
  proxies like Cloudflare AI Gateway with no provider key in the
  environment.
- `Compiler.compile()` accepts `extra_instructions`, appended to the
  compiler skill prompt (e.g. pinning emitted judge calls to a
  specific runtime client).
- The Cloudflare adapter emits `manifest.json` at the runtime-dir
  root — machine-readable pipeline identity (name, version, pipeline
  hash, class name, node ids, entry) for deploy tooling; regex
  fallback retained for older emits.
- `rote.eval.build_scorecard_for` is now public API (was a CLI
  private).
- **Cloudflare Workers call MCP tools, authenticated by provisioning**
  — Workers have no filesystem for the token store, so
  `rote mcp export <server>` turns a completed `rote mcp login` into
  Worker secrets (dotenv form for `.dev.vars`, `--json` for
  `npx wrangler secret bulk`). MCP-bound nodes emit working calls
  through a Workers helper that mints access tokens at runtime via the
  OAuth refresh grant and caches them — with rotated refresh tokens —
  in a `ROTE_MCP_TOKENS` KV namespace declared in the emitted
  `wrangler.jsonc`. The Env interface, `.dev.vars.example`, and
  README document the per-server provisioning surface. Emitted output
  typechecks against `@cloudflare/workers-types` v5 + the MCP SDK.

- **Node TypeScript runtimes call MCP tools, authenticated** — the
  DBOS-TS and Inngest adapters now emit *working* bodies for
  `mcp:`-bound nodes (previously always throwing stubs): the module
  calls the tool via the official `@modelcontextprotocol/sdk` (^1.29)
  through an emitted `src/extracted/_roteMcp.ts` helper that reads the
  same rote registry/token store the CLI writes, refreshes stale access
  tokens with a refresh-token grant against the stored
  `token_endpoint`, writes rotated refresh tokens back atomically, and
  retries once on 401. `--backend api` keeps the direct-vendor-SDK
  stubs. Proven end-to-end across languages: Python `rote mcp login`
  seeds the store, compiled TS authenticates, refreshes a forced-stale
  token, and rotates credentials Python reads back
  (`tests/test_mcp_ts_e2e.py`).
- **A full MCP client with OAuth 2.1** (`rote mcp`, design in
  [docs/mcp-client.md](docs/mcp-client.md)) — real streamable-HTTP MCP
  servers authenticate with OAuth; without a client that can run the
  flow, store tokens durably, and refresh them, MCP-backed workflows
  only worked against unauthenticated servers:
  - `rote mcp add / list / remove` — a user-level server registry
    (`~/.config/rote/mcp.json`, 0600) mapping the logical server names
    pipelines carry in their `mcp:` bindings to endpoints, with
    pre-registered `client_id`/`client_secret` support for servers
    without dynamic client registration (Slack/GitHub-class) and static
    headers for API-key schemes.
  - `rote mcp login` — the full spec dance (protected-resource
    discovery, PKCE, dynamic client registration, refresh) via
    fastmcp's OAuth provider, persisting into a rote-owned token store
    (one 0600 JSON file per server, `~/.local/share/rote/mcp-tokens/`)
    whose layout is a documented cross-language contract.
    `--no-browser` prints the authorization URL for SSH boxes.
  - `rote mcp headers` — mints a currently-valid Authorization header
    (auto-refreshing through the stored refresh token), the
    machine-facing token API.
  - **Emitted DBOS apps authenticate**: MCP-backed steps now open the
    client through an emitted `extracted/_rote_mcp.py` helper (the
    verbatim source of `rote.mcp._runtime_helper` — one tested
    implementation) that resolves endpoints (env > registry > IR) and
    credentials (OAuth store > static headers > none) at runtime, with
    in-place token refresh. Emitted apps still never import rote.
  - **`eval --run` trials authenticate**: the generated `--mcp-config`
    injects registry headers verbatim, or — for logged-in servers — a
    `headersHelper` invoking `rote mcp headers`, which Claude Code
    re-runs per connection and on 401, so tokens refresh mid-run on
    long trials.
  - Verified end-to-end against a real OAuth-protected server
    (`tests/test_mcp_oauth_e2e.py`): live authorization dance with
    dynamic registration, cross-process token reuse, an authenticated
    tool call through the emitted helper, and a forced-stale refresh.
  - New optional extra: `pip install 'rote-cli[mcp]'`; `python -m rote`
    now works (used by the headersHelper wiring).

### Fixed
- The API drivers no longer misread per-turn output truncation
  (`stop_reason: max_tokens` / `finish_reason: length`) as completion —
  the per-turn cap is raised to 32k tokens and the loop continues
  automatically with a `warning` event. Surfaced by Claude 5 models,
  which spend far more of the turn budget on thinking.
- The Anthropic driver sets an explicit client timeout sized from the
  per-turn token cap (the SDK otherwise refuses non-streaming requests
  that may exceed 10 minutes at the new cap).
- The Anthropic driver survives `content: null` assistant turns some
  gateway endpoints return for all-thinking responses: content is
  normalized, empty assistant turns are never replayed into history,
  and an empty natural stop nudge-continues with a warning.
- Emitted Cloudflare `package.json` pinned `@cloudflare/workers-types`
  ^4, which current wrangler (4.110+) rejects with a peer-dependency
  conflict on a fresh install — bumped to ^5.

## [0.8.0] - 2026-07-10

### Added
- **`examples/invoice-push/`** — the fourth committed example and the
  `agent_loop` archetype, completing node-kind coverage across the
  examples (adapted from a real production browser-automation skill,
  fully fictionalized): a bounded per-row browser cycle (5-node
  `loop_body`, termination cap) that cannot crystallize into code,
  surrounded by 12 deterministic nodes. First example to commit its
  `eval.yaml` sidecar — with per-row `iterations`, the calibration
  fixture for the loop-aware cost model below.
- **`eval --run` auto-wires the pipeline's MCP servers into the skill
  trial.** The compiled pipeline's `mcp:` bindings name exactly the
  servers the source skill uses, so the "before" measurement now runs
  the agent over the same live tools (`--mcp-config` +
  `--strict-mcp-config`, per-server `mcp__<server>__*` allowlists;
  URLs resolve from the binding's explicit `url`, else
  `ROTE_MCP_<SERVER>_URL` — the adapters' rule). Verified live against
  a real `claude -p` run over a Streamable-HTTP MCP server.
- **Reliability flags on measured skill runs.** Each trial is checked
  structurally (never by an LLM) before it may calibrate priors:
  `errored`, `hit_max_turns` (truncated — its cost is a floor, not a
  measurement), `suspiciously_few_turns` (fewer turns than the pipeline
  has data pulls — the agent never did the work), and
  `missing_mcp_servers` (the skill ran without its tools). Flagged runs
  are excluded from `suggested_priors` re-fits, listed in the measured
  scorecard section, and recorded in the calibration corpus with their
  flags.
- **Loop-aware before-cost model**, calibrated against a production
  browser-automation skill whose two real runs measured 184 and 730
  agent turns (~$8.6 and ~$32 on Sonnet) against a prior estimate of
  $0.82–$2.75:
  - The eval sidecar's `StepEstimate` gains `iterations: {low, high}` —
    a step that repeats per row/page/item declares its per-iteration
    turns and realistic repeat count, and the whole-run turn estimate
    multiplies them. Loop-dominated skills were previously understated
    5–20× because the schema could not express iteration at all (the
    compiler's own notes described the loop economics correctly; the
    form had nowhere to put the number).
  - New `transcript_cap_tokens` prior (default 165k, the per-turn
    cache-read plateau measured on both production runs): the transcript
    an agent re-reads saturates at the harness's compaction ceiling
    instead of growing without bound, so cached-read totals transition
    from quadratic to linear on long runs. Without the cap, correcting
    the turn count would have swung the error the other way (~2.5×
    over).
  - The compiler rubric's calibration anchors are now regime-aware:
    sequential tool-heavy skills anchor at 30–57 turns, per-item loop
    skills at hundreds (the old universal 30–57 anchor actively pulled
    loop-skill estimates down an order of magnitude).

  Re-estimated with both fixes, the calibration skill's scorecard
  brackets reality: $1.96–$22.68 estimated vs $8.59–$32.00 measured
  (Sonnet), 98M estimated token ceiling vs 96M measured.

### Fixed
- `rote compile` now re-points `eval.yaml`'s `source_skill` alongside
  `pipeline.yaml`'s — the sidecar previously kept the agent's
  temp-work-dir-relative path, a dead pointer in every kept compilation.

## [0.7.0] - 2026-07-09

### Added
- **Two production-shaped examples** alongside BDR, each adapted from a
  real production skill with all identifiers fictionalized:
  `examples/ops-report/` (the 100%-roteness archetype — every step
  deterministic, one durable HITL gate, zero LLM nodes; the fixture for
  the `python` adapter's durable-execution refusal) and
  `examples/deal-monitor/` (the data-heavy archetype — parallel entry
  waves, fan-out judges, template render replacing LLM-generated HTML;
  the calibration fixture for the payload-aware estimator).
  `tests/test_examples.py` guards every example's expected IR, including
  that its `source_skill` pointer resolves.
- **Payload-aware "before" cost estimator** — the static scorecard now
  models the data a skill pulls into context, not just its turn count.
  The compiled pipeline's `external_call` footprint sizes the agent-side
  context payload (`tokens_per_external_call_result`, default 6k/call,
  with a per-MCP-tool override table `payload_tokens_per_tool`), folded
  into C₀ of the cache-aware transcript model. Calibrated against a real
  data-heavy production skill (Slack + Gmail dashboard, ~22 turns,
  ~1.6M cache-read tokens/run): the old flat prior underestimated the
  before-cost 5–15×; the payload-aware default lands within ~3×, and one
  `eval --run` calibration brings it within ~10%.
- `rote eval --run`'s suggested prior re-fits now include
  `transcript_growth_per_turn`, inverted from measured cache-read tokens
  under the quadratic transcript model — so every empirical run reports
  the effective payload-inclusive growth rate alongside
  `seconds_per_turn` and `output_tokens_per_turn`.
- **`CodexDriver` is now implemented** — `rote compile --agent codex`
  spawns `codex exec` (OpenAI Codex CLI) as a compiler backend,
  completing the three-driver lineup. Runs headless under a
  `workspace-write` sandbox (global reads so it can read the skill +
  rubric in place, writes confined to the work dir, no network). The
  environment is passed through untouched — a stored `codex login`
  session is only overridden by `CODEX_API_KEY`/`CODEX_ACCESS_TOKEN`,
  not `OPENAI_API_KEY`, and there is no separate OpenAI-API driver, so
  no auth is forced. Verified against the real CLI (codex-cli 0.142.4).
- **`rote analyze` is now implemented** — the `plan` to `compile`'s
  `apply`. Runs the compiler against a skill and prints a structural
  report (node-kind breakdown, roteness — matching `rote eval` — plus
  mandatory checks, HITL gates, agent loops, and which runtimes can
  target it) *without* emitting runtime code. `--json` for a
  machine-readable report; `--out` keeps the compiled IR for a later
  `rote emit`. Previously a stub that printed "not yet implemented".

### Changed
- **`source_skill` no longer participates in the pipeline hash** — it's
  provenance (a filesystem path the compiler re-points per output
  location), and hashing it minted a new workflow type on every
  re-compilation to a different directory, re-versioning in-flight
  workflows whose behavior didn't change. Same rule as `Node.source`,
  which was already excluded. **One-time consequence:** every pipeline's
  hash (and therefore emitted workflow type name) changes once with this
  release; in-flight workflows on the old type names continue on old
  code as designed.

### Fixed
- The `codex` driver no longer raises `NotImplementedError` when
  selected via `--agent codex` or chosen by auto-detect (a first-run
  crash for users who had the Codex CLI but not Claude Code installed).
- The BDR example's committed IR baseline carried the same dead
  `source_skill` pointer the orchestrator fix addresses
  (`../../skill` resolved to a nonexistent `examples/skill`), so
  `rote eval` on the canonical example silently dropped its before-side
  baseline. Corrected to `../skill`; now regression-guarded for every
  example.
- `rote compile`/`rote analyze --out` now re-point the pipeline's
  `source_skill` to resolve from the emitted `pipeline.yaml`'s location
  (relative when possible, absolute otherwise). The agent records the
  path relative to its temp work dir — deleted when the run ends — so
  every kept compilation carried a dead pointer and a later `rote eval`
  silently dropped the entire before-side baseline ("source_skill did
  not resolve — emitting the after-side only"). Found by compiling a
  real production skill.

## [0.6.0] - 2026-07-05

### Added
- **Workers AI signature client** (`signature_spec.client: "workers-ai"`): the
  Cloudflare adapter emits `env.AI.run(...)` with schema-locked JSON output
  (`response_format: json_schema`) routed through an AI Gateway — no vendor
  SDK and no API key (the `AI` binding is the auth). `Env` / `wrangler.jsonc` /
  secrets are now client-aware: the `AI` binding appears when a judge targets
  workers-ai, and only the API keys actually used are emitted.
- **Roteness** in the eval scorecard: `deterministic steps / total steps`, a
  purely structural code-vs-inference ratio (0% = agent loop, 100% = pure
  code) that never depends on a model estimate — the honest counterweight to
  the empirical determinism metric. On `SamplingSurface.roteness`,
  `Scorecard.to_dict()`, and the markdown scorecard.
- MCP-backed `external_call` nodes: an IR `mcp:` binding (server / tool /
  args / url / transport) lets a compiled pipeline **call the MCP tool
  the source skill used, over Streamable HTTP**, instead of emitting a
  `NotImplementedError` stub — so the output runs out of the box. The DBOS
  adapter emits a working FastMCP client call by default
  (`external_backend="mcp"`); `api` falls back to the direct-SDK `impl`.
  Verified end-to-end against a live mock MCP server (`tests/test_mcp_e2e.py`).
- `rote emit`/`rote compile` gained `--backend mcp|api` to choose that
  backend at emit time (adapter factories now accept forwarded options).
- `rote eval` harness: an auto-emitted `scorecard.md` with a static
  before/after estimate of speed, cost, and determinism (live-fetched
  model prices, no hardcoded tables), plus `rote eval --run` for
  empirical trials of both the source skill and the emitted pipeline.
- Open-source project scaffolding: CI on every pull request and push
  (Python 3.11–3.13 matrix + lint/type/sanity gates), `SECURITY.md`,
  `CODE_OF_CONDUCT.md`, this changelog, issue/PR templates, Dependabot,
  and a pre-commit config.

### Fixed
- Corrected `pip install rote[...]` → `rote-cli[...]` across the docs
  (the bare `rote` name is an unrelated PyPI package).
- Refreshed stale version/adapter/test-count references in the README,
  `CONTRIBUTING.md`, and the BDR example README.

## [0.5.0]

### Added
- **Inngest adapter** (`inngest`): emits a TypeScript Inngest app that
  mounts into an existing Node/Next.js service, with `waitForEvent`
  HITL gates.
- **Raw Python adapter** (`python`): a maximum-legibility,
  orchestrator-free plain-script target; refuses `hitl_gate` pipelines
  at emit time.
- DBOS trigger backend for `rote serve`, so the default runtime can be
  MCP-triggered.

## [0.4.0]

### Added
- **DBOS TypeScript adapter** (`dbos-ts`): a zero-infrastructure
  durable target for Node shops (shares `_ts_common` with Cloudflare).

### Changed
- **DBOS is now the default runtime.**

### Security
- Hardened the IR and every emitter against code injection from crafted
  `pipeline.yaml` input (charset-constrained id/signal/impl fields at
  the IR boundary; escaped prose at emission).

## [0.3.0]

### Added
- **Cloudflare Workflows adapter** (`cloudflare`) — the second runtime,
  proving the IR is runtime-agnostic — and the **DBOS adapter**.
- `rote serve` / `rote register`: expose compiled pipelines as MCP
  tools (FastMCP 3.x, stdio + Streamable HTTP).
- Data-flow threading: nodes declare `inputs:` and adapters thread real
  payloads through the DAG.
- `rote` shipped as a Claude Code plugin.

### Changed
- Distribution renamed to `rote-cli` on PyPI (import and CLI stay
  `rote`); package made release-ready with tag-driven Trusted
  Publishing.

## [0.1.0]

### Added
- Initial release: the runtime-agnostic IR (`pipeline.yaml`), the
  Temporal adapter, the compiler agent + driver layer (Claude,
  Anthropic API, Codex stub), the `rote compile` / `rote emit` CLI,
  and the BDR-outreach example skill.

[Unreleased]: https://github.com/trevhud/rote/compare/v0.15.0...HEAD
[0.15.0]: https://github.com/trevhud/rote/compare/v0.14.0...v0.15.0
[0.14.0]: https://github.com/trevhud/rote/compare/v0.13.0...v0.14.0
[0.13.0]: https://github.com/trevhud/rote/compare/v0.12.1...v0.13.0
[0.12.1]: https://github.com/trevhud/rote/compare/v0.12.0...v0.12.1
[0.12.0]: https://github.com/trevhud/rote/compare/v0.11.0...v0.12.0
[0.11.0]: https://github.com/trevhud/rote/compare/v0.10.0...v0.11.0
[0.10.0]: https://github.com/trevhud/rote/compare/v0.9.0...v0.10.0
[0.9.0]: https://github.com/trevhud/rote/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/trevhud/rote/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/trevhud/rote/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/trevhud/rote/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/trevhud/rote/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/trevhud/rote/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/trevhud/rote/releases/tag/v0.3.0
[0.1.0]: https://github.com/trevhud/rote/releases/tag/v0.1.0
