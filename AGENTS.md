# AGENTS.md — driving `rote` as an installed tool

This file is for a **coding agent** (Codex, Cursor, a plain API agent, or
Claude Code without the plugin) that has been asked to **use** `rote` to
graduate a skill. It covers the operational facts `rote --help` can't tell
you. If you are editing `rote` itself, read `CLAUDE.md` instead — that's the
contributor manual and does not apply here.

`rote` compiles an Anthropic-style skill (`SKILL.md` + optional
`references/`) into a deterministic, durable workflow: a `pipeline.yaml` IR
plus runnable runtime code. The north-star command is `rote graduate`.

## Invocation contract

Run it with `uvx` — nothing to install but `uv`:

```sh
uvx --from rote-cli rote <args>
```

- The PyPI **package** is `rote-cli`; the **command** (and import name) is
  `rote`. The published wheel ships no `rote-cli` executable, so
  `uvx rote-cli ...` fails. Always `--from rote-cli rote`.
- Unreleased features: swap the origin, same command —
  `uvx --from git+https://github.com/trevhud/rote rote <args>`.
- Do not clone or build a venv; `uvx` handles isolation.

## Preflight first: `rote doctor`

Before spending money on a graduation, run the read-only preflight — it costs
nothing and tells you whether a run can even succeed:

```sh
uvx --from rote-cli rote doctor --json
```

`--json` gives `{version, python, drivers:[{name,available,reason}], runtimes,
mcp_servers:[{name,url,auth}], apps, ok}`. Gate on `ok` (true iff at least one
graduator driver is available — `rote doctor` also exits non-zero when none is).
Each unavailable driver's `reason` is an actionable fix. It also surfaces MCP
servers whose auth is `expired`/`not authenticated` (those would park a run) and
registered apps whose directory has gone missing. It does NOT probe CLI
subscription auth for claude/codex — that's only verified at run time.

## Optional but recommended: `rote baseline` before graduating

```sh
uvx --from rote-cli rote baseline <skill-dir> --out <out-dir> --input task.json --json
```

One instrumented run of the *raw* skill (via `claude -p`, subscription-billed,
minutes + tens of cents) that produces three things at once under
`<out-dir>/baseline/`: measured before-metrics (`metrics.json` — wall clock,
turns, tokens, cost), the full transcript per trial, and
`observed-tools.json` — every MCP tool the agent actually called with its
real input/result payloads (ground truth for which servers the skill needs).
`--input` is the skill's task payload as a JSON object. **Omit it** and rote
derives a representative input from SKILL.md (cheap single-shot call) and
asks for confirmation before spending; as an agent you are not on a TTY, so
either pass `--input` (preferred — confirm the payload with the human first)
or `--yes` to accept the derived proposal. The proposal is always saved to
`baseline/derived-input.json` for editing. rote's registered MCP servers are
injected automatically (`rote mcp login` covers this run too). **Side
effects are gated**: only tools whose server declares `readOnlyHint` are
callable; pass `--allow-writes` only after the human confirms this skill's
writes may fire once. `--json` mirrors the metrics plus `observed_servers`
and per-server skip reasons.

`rote graduate --baseline` runs the same measurement immediately before
graduating (flags: `--baseline-input`, `--yes`, `--allow-writes`) — the
scorecard then gains a **Measured baseline** section and the `--json`
payload a `baseline` object. Without `--baseline`, graduate prints a
one-line tip recommending it; it never blocks.

## `rote graduate` is slow and costs money — plan for it

```sh
uvx --from rote-cli rote graduate <skill-dir> --runtime <runtime> --out <out-dir>
```

- A realistic skill takes **~13 minutes** and 30–40 agent turns (~$0.70 on a
  Claude subscription). **Run it backgrounded** and poll — do not block.
- **Confirm the resolved skill directory as an absolute path** with the human
  before launching. Graduation spends real time and tokens; never guess-and-go.
  If the directory has no `SKILL.md`, stop and ask.
- Progress streams to **stderr**, one line each (stdout is reserved for the
  final summary). Tail it to know where the run is:
  - `[phase N/7] <name>` — phase transitions.
  - `[turn N] <tool> <path>` — the agent's tool calls.
- `--runtime` defaults to `dbos` (durable execution as a library, no
  orchestrator, SQLite for dev). Others: `temporal`, `python`, `cloudflare`,
  `dbos-ts`, `inngest`. `--out` is required.

## Auth: do not export an API key to "fix" it

The default `claude` driver spawns `claude -p` and **deliberately scrubs
`ANTHROPIC_API_KEY` and `ANTHROPIC_AUTH_TOKEN`** from the child environment so
the run bills against the user's Claude Max/Pro subscription, not per-token
API charges. If a run seems to want auth, that is expected — do **not** set an
API key to work around it.

- Want API-key billing on purpose? Pass `--agent api` (the in-process
  Anthropic-SDK driver, which reads `ANTHROPIC_API_KEY`).
- `--agent` choices: `claude` (default), `codex` (ChatGPT subscription),
  `api`. Auto-detect order when omitted: `claude` → `codex` → `api`.

## Failure recovery: check for the file before trusting the exit code

The graduator's deliverable is a file on disk. If a `rote graduate` run exits
nonzero, **check whether `<out-dir>/graduated/pipeline.yaml` exists anyway** —
a subprocess blip *after* the agent wrote a complete IR still produces valid
output, and the run has materially succeeded. Treat the file's presence as the
real success signal; surface the stderr warning either way. Do not re-run a
graduation (and re-spend ~$0.70) on an exit code alone.

## Output layout

`rote graduate --out <dir>` writes two trees:

- `<dir>/graduated/` — the agent's artifacts:
  - `pipeline.yaml` — the validated IR (source of truth).
  - `extracted/*.py` — **stubs you implement** (see below).
  - `signatures/*.py` — complete, typed LLM judges. **Leave these alone.**
  - `scorecard.md`, `graduation-report.md` — before/after estimate + report.
- `<dir>/runtime/<runtime>/` — the deployable code: `main.py` (DBOS/Python),
  `workflow.py` + `activities.py` (Temporal), or `src/workflow.ts` +
  `wrangler.jsonc` (Cloudflare/TS), plus its own `README.md` on how to run,
  signal HITL gates, and deploy. Read that README before deploying.

`rote emit <pipeline.yaml> --out <dir>` (the pure IR→code step, no LLM, no
cost) writes the runtime tree directly into `<dir>`.

## The job you finish: implement `extracted/*`, not `signatures/*`

After graduation the pipeline is not yet runnable. The deterministic nodes are
scaffolds that `raise NotImplementedError`:

- **`extracted/*.py`** — implement each function body with the real vendor API
  call (the source skill's MCP tool calls were graduated away). The step in
  `main.py` calls these with the node's payload as keyword arguments; to learn
  a stub's exact input/output contract, read its call site in `main.py` and
  the node's `inputs:`/`output:` in `pipeline.yaml`.
- **`signatures/*.py`** — already complete typed LLM judges (Pydantic
  input/output, embedded schema, prompt). Do not rewrite them.

To run a graduated pipeline locally once stubs are filled:
`rote run <out-dir> --input '{"your": "input"}'` — it resolves the
emitted runtime (`--runtime` disambiguates when several were emitted),
executes `python`/`dbos` in-process, and delivers HITL gate payloads
passed as `--signal <gate>='{...}'` (non-interactive runs **must** pass
one per gate or the command exits 2 listing them). The run's output JSON
is stdout; status is stderr. `rote run <skill-dir>` runs the *raw skill*
the same way `rote baseline` does (read-only MCP gate, `--allow-writes`
to lift), without writing baseline artifacts. Equivalent manual form for
a DBOS app: `python main.py '{"your": "input"}'` (see the runtime
`README.md`).

## The graduate default follows the login state

If the machine has a rote-cloud login (`rote login`; credential at
`~/.local/share/rote/cloud.json`), `rote graduate` **without an explicit
`--runtime` defaults to `cloudflare` and auto-deploys the emitted app to
rote cloud** after emission. Consequences for automation:

- Pass `--runtime <x>` explicitly (or `--no-deploy`) if you need
  deterministic local-only behavior regardless of login state.
- A deploy failure exits 1 but the local artifacts are complete — retry
  with `rote deploy <out-dir> --target rote-cloud`, don't re-graduate.
- Logged out, nothing changes: local `dbos` default, plus a one-line
  stderr hint. There is never an interactive prompt on this path.
- `rote whoami --json` → `{url, user, tenant}` (exit 1 when not logged
  in or the credential is rejected) is the cheap way to probe the state.

## Re-running is safe — retry freely

`rote` never silently clobbers your edits.

- Every emit records what it wrote in `.rote-manifest.json`. On re-emit, any
  file you edited since is **left untouched**; the fresh version lands beside
  it as `<name>.new` and the CLI reports it. Merge or delete `.new` files.
- `rote graduate --update` is **incremental**: it diffs the skill against the
  previous run's `provenance.json`, re-derives only nodes whose source
  sections changed, keeps unchanged nodes' ids (so in-flight durable workflows
  aren't orphaned), and preserves stubs you've filled. No section changes → no
  agent run at all (no cost).

A naive retry of `emit` or `graduate --update` is therefore non-destructive.

## Exit codes

- `0` — success.
- `1` — runtime error (bad pipeline, graduator failure, price fetch, etc.).
- `2` — usage error (bad flag, missing path, unknown runtime).
- `130` — interrupted (Ctrl-C).

## Machine-readable output

Pass `--json` to get structured output instead of prose:

- `rote analyze <skill> --json` — node-kind breakdown, roteness, HITL gates,
  targetable runtimes (dry-run; no runtime code emitted).
- `rote eval <graduated> --json` — the before/after scorecard as JSON.
- `rote mcp list --json` — registered MCP servers + auth status.
- `rote emit --json` — the final result object: pipeline name/version,
  `runtime`, `out_dir`, `written` (label → absolute path), `preserved_new_files`
  (`.new` siblings), `unimplemented_stubs` (the `extracted/*` files still
  needing an implementation — your TODO list), and `mcp_servers` (see below).
  Parse this instead of scraping the human summary.
- `rote graduate --json` — a superset of the emit object that also carries
  `graduated_dir`, `runtime_dir`, `scorecard` (or `null` under `--no-eval` /
  a price-fetch failure), the `driver` used, and — when the logged-in
  auto-deploy ran — a `deploy` object (`target`, `ok`, `detail`/`error`).
- `rote deploy --json` — one deploy report: `target`, `runtime`, `app_dir`,
  `ok`, `action` (`deployed`/`dry-run`/`guidance`), `detail`. Guidance
  runtimes (temporal/inngest/python) exit 0 with `action: "guidance"`.
- `rote run --json` — one result object: `kind` (`skill`/`pipeline`), the
  run's `output`, and side-specific measurement (`run` metrics +
  `observed_servers` for skills; `runtime`, `wall_seconds`, `error`,
  `judge_usage` for pipelines). Without `--json`, stdout is the bare
  output JSON.

`mcp_servers` (on `analyze`, `emit`, `graduate`, and the progress-file
summary line) is the pipeline's MCP requirements manifest: one entry per
required server — `{server, nodes, tools, auth}` — derived from the IR's
`mcp:` bindings and joined against the local registry/token store. `auth`
is the five-state value doctor uses, plus `"not registered"` when the
server isn't in the rote registry at all. This is **advisory, never
blocking**: an unauthenticated server won't fail a run (it parks durably
and `rote mcp login <server>` releases it), but resolving the listed
recommendations *before* running avoids mid-flight parks. Empty list means
the pipeline makes no MCP calls.

Note: `unimplemented_stubs` is read off the emitted `extracted/*` files, so it
is populated for the `dbos` (default), `python`, `cloudflare`, `dbos-ts`, and
`inngest` runtimes; `temporal` inlines its stubs in `activities.py` and reports
an empty list.

Exit codes above let you distinguish success from failure without parsing.

### Watching a `graduate` run live: `--progress-file`

`graduate` runs for ~13 minutes; `--json` only prints once, at the end. To
see what it's doing *as it happens*, pass `--progress-file PATH` — rote streams
one JSON object per line (NDJSON), flushed live, that you tail while the run
proceeds. It runs alongside the human stderr log and composes with `--json`.

```sh
uvx --from rote-cli rote graduate <skill> --out <dir> --progress-file run.ndjson &
tail -f run.ndjson        # each line is a complete JSON object
```

Event lines (`None` fields omitted):

```jsonc
{"type":"phase","ts":..,"phase":2,"phase_name":"Node Classification"}
{"type":"turn","ts":..,"turn":12,"tokens":{"input":40234,"output":8100},"cost_usd":0.24}
{"type":"tool","ts":..,"turn":12,"tool_name":"Write","path":"signatures/qualify.py"}
{"type":"complete","ts":..,"message":"..."}
```

- `tokens` is **cumulative** run spend; `cost_usd` is priced live from the
  model's current published rates (omitted if pricing is offline — never fails
  the run). This works on the default `claude` driver and `--agent api`; the
  `codex` driver emits phase lines only.
- The **last line** is a `type:"summary"` digest — the machine end-of-run
  report: `roteness`, `node_kinds` (counts by kind), `nodes`, `graduated_dir`,
  `runtime_dir`, `unimplemented_stubs`, `total_tokens`, `cost_usd`. Read this
  instead of running `rote analyze` again. `roteness`/`node_kinds` are
  terminal (they need the finished pipeline), so they appear only here, not in
  the per-turn stream.

## MCP-authenticated steps: the park-on-auth loop

If a graduated node calls an MCP server that needs credentials, the emitted
workflow **parks durably** on a missing/dead token rather than failing.
Authenticate the server (`rote mcp login <server>`) and every parked run is
released automatically. Full details, including releasing runs when the
credential was fixed another way (`rote mcp release <server>`), are in
[`docs/mcp-client.md`](docs/mcp-client.md).

To expose a deployed pipeline back to an agent as an MCP tool, see
`rote register` + `rote serve` and [`docs/mcp-trigger.md`](docs/mcp-trigger.md).
