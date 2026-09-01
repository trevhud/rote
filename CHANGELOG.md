# Changelog

All notable changes to `rote` (distributed on PyPI as
[`rote-cli`](https://pypi.org/project/rote-cli/)) are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While `rote` is pre-1.0, minor versions may include breaking changes.

## [Unreleased]

### Added
- **Heartbeat events while awaiting a model request.** The api and
  openai-api drivers emit a `log` event every 120 seconds while an LLM
  request is pending, so a hosted hang detector can tell a slow or
  retrying request from a wedged job.
- **The api and openai-api compiler drivers can call MCP tools live.**
  A new driver kwarg `mcp_servers` (each entry `{"name", "url",
  "headers"}`, streamable HTTP only) connects at agent start and exposes
  each server's tools to the compiler agent as `mcp__<server>__<tool>` —
  gated to tools whose server declares `readOnlyHint`, the same
  predicate `rote baseline` already applies, now shared in
  `rote.mcp.live_tools` so the two gates cannot drift. Connections stay
  open across the agent loop; a server that cannot be reached is a
  warning event, never a failed compile, and a tool failure returns to
  the model as an error tool result. When `rote compile` runs one of
  these drivers and no servers were passed, the local registry's
  authenticated servers are wired in automatically (static headers
  verbatim, logged-in servers via a freshly refreshed token;
  unauthenticated servers are skipped with a printed reason).
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
- **Pinned `fastmcp` below 4.** The declaration was `fastmcp>=3.4.2` with
  no upper bound, so CI resolved to `fastmcp` 4.0.2 (published
  2026-09-02) and `mcp` 2.x on every fresh install while local
  environments stayed on 3.x. Five tests across three CI jobs began
  failing on an unchanged `main`: the `serve` client stops receiving
  tool-list-changed notifications, two OAuth e2e tests fail, and the TS
  runtime e2e reports `Method not found`. The pin restores a trustworthy
  signal; migrating to the 4.x / 2.x line is separate work, and MCP SDK
  v2 also renames `Tool.inputSchema` to `Tool.input_schema`.


### Fixed
- **A compilation's reported tokens and cost ignored prompt caching.**
  `claude -p` always runs prompt-cached, so `input_tokens` holds only the
  handful of tokens that were neither written to nor read from the cache.
  The driver summed that field alone, and the CLI priced the result: a
  22-minute, 62-turn compilation was reported as `tokens in=113 out=1919;
  cost $0.029`. All four token buckets are now counted, and every display
  adds cache writes and reads into the input figure it shows.
- **The agent runtime's own billed cost was discarded.** The driver read
  `result["cost_usd"]`, a key the `claude -p` result envelope does not
  have. The field is `total_cost_usd`, so the read returned `None` on
  every run while the authoritative number sat one key away. It is now
  read, and it outranks any cost priced locally, because the CLI knows
  the per-model split and the exact cache rates that applied.
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

[Unreleased]: https://github.com/trevhud/rote/compare/v0.13.0...HEAD
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
