# Changelog

All notable changes to `rote` (distributed on PyPI as
[`rote-cli`](https://pypi.org/project/rote-cli/)) are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While `rote` is pre-1.0, minor versions may include breaking changes.

## [Unreleased]

### Added
- **Payload-aware "before" cost estimator** — the static scorecard now
  models the data a skill pulls into context, not just its turn count.
  The graduated pipeline's `external_call` footprint sizes the agent-side
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
- **`CodexDriver` is now implemented** — `rote graduate --agent codex`
  spawns `codex exec` (OpenAI Codex CLI) as a graduator backend,
  completing the three-driver lineup. Runs headless under a
  `workspace-write` sandbox (global reads so it can read the skill +
  rubric in place, writes confined to the work dir, no network). The
  environment is passed through untouched — a stored `codex login`
  session is only overridden by `CODEX_API_KEY`/`CODEX_ACCESS_TOKEN`,
  not `OPENAI_API_KEY`, and there is no separate OpenAI-API driver, so
  no auth is forced. Verified against the real CLI (codex-cli 0.142.4).
- **`rote analyze` is now implemented** — the `plan` to `graduate`'s
  `apply`. Runs the graduator against a skill and prints a structural
  report (node-kind breakdown, roteness — matching `rote eval` — plus
  mandatory checks, HITL gates, agent loops, and which runtimes can
  target it) *without* emitting runtime code. `--json` for a
  machine-readable report; `--out` keeps the graduated IR for a later
  `rote emit`. Previously a stub that printed "not yet implemented".

### Fixed
- The `codex` driver no longer raises `NotImplementedError` when
  selected via `--agent codex` or chosen by auto-detect (a first-run
  crash for users who had the Codex CLI but not Claude Code installed).

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
  args / url / transport) lets a graduated pipeline **call the MCP tool
  the source skill used, over Streamable HTTP**, instead of emitting a
  `NotImplementedError` stub — so the output runs out of the box. The DBOS
  adapter emits a working FastMCP client call by default
  (`external_backend="mcp"`); `api` falls back to the direct-SDK `impl`.
  Verified end-to-end against a live mock MCP server (`tests/test_mcp_e2e.py`).
- `rote emit`/`rote graduate` gained `--backend mcp|api` to choose that
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
- `rote serve` / `rote register`: expose graduated pipelines as MCP
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
  Temporal adapter, the graduator agent + driver layer (Claude,
  Anthropic API, Codex stub), the `rote graduate` / `rote emit` CLI,
  and the BDR-outreach example skill.

[Unreleased]: https://github.com/trevhud/rote/compare/v0.6.0...HEAD
[0.6.0]: https://github.com/trevhud/rote/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/trevhud/rote/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/trevhud/rote/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/trevhud/rote/releases/tag/v0.3.0
[0.1.0]: https://github.com/trevhud/rote/releases/tag/v0.1.0
