---
name: serve
description: >-
  Wire a compiled rote pipeline up as an MCP tool so Claude can trigger the
  deployed workflow directly. Use when the user says "register my compiled
  pipeline", "serve my pipelines over MCP", "trigger the workflow from
  Claude", "hook the pipeline up to Claude", or asks what to do after
  `rote compile` and deployment. Covers `rote register` and `rote serve`
  plus the `claude mcp add` wiring.
---

# Serve compiled pipelines as MCP tools

`rote serve` is one MCP server exposing every registered pipeline as a
callable tool. The flow:

```
rote compile → deploy the runtime → rote register → rote serve → call from Claude
```

`rote serve` **triggers deployed workflows; it does not host them.**
MCP triggering supports the `dbos` (default), `temporal`, and
`cloudflare` runtimes.

The CLI ships on PyPI as `rote-cli` with an executable named `rote`,
so every invocation is `uvx --from rote-cli rote <args>` (never
`uvx rote-cli ...`). For unreleased features, substitute the source:
`uvx --from git+https://github.com/trevhud/rote rote <args>`.

## 1. Check preconditions

- A compile output directory exists (contains `compiled/pipeline.yaml`).
- The runtime side is running: for DBOS, the emitted app in worker mode
  (`python main.py --serve` or `dbos start`) against the system
  database you'll register — enqueued runs sit in status `enqueued`
  until that process exists; for Temporal, a worker against the user's
  cluster; for Cloudflare, `wrangler deploy` done. If not, stop and
  help with that first.

## 2. Register the pipeline

```sh
# DBOS (the default). System DB URL: --system-database-url, else
# $DBOS_SYSTEM_DATABASE_URL, else the emitted app's SQLite file
# (derived from <out-dir>/runtime/dbos/main.py).
uvx --from rote-cli rote register <out-dir>

# Temporal (defaults: localhost:7233, namespace "default",
# task queue = pipeline.name, workflow type = the emitted versioned name)
uvx --from rote-cli rote register <out-dir> --runtime temporal

# Cloudflare
uvx --from rote-cli rote register <out-dir> --runtime cloudflare \
  --url https://<worker>.workers.dev
```

This upserts `~/.rote/registry.json`. Re-registering updates in place.
**After re-compiling a changed skill, register again** — the DBOS and
Temporal workflow names are derived from the pipeline content hash and
must stay in sync with the emitted code.

## 3. Add the MCP server to Claude

`rote serve` needs the `serve` extra (FastMCP) plus `dbos` when any
registered pipeline runs on DBOS, so the spec includes both:

```sh
claude mcp add --scope user rote -- uvx --from 'rote-cli[serve,dbos]' rote serve
```

For unreleased features, use the GitHub source instead:

```sh
claude mcp add --scope user rote -- \
  uvx --from 'rote-cli[serve,dbos] @ git+https://github.com/trevhud/rote' rote serve
```

Verify with `claude mcp list`. Each registry entry becomes two tools
(three for DBOS): `<name>` (starts a run, returns `{workflow_id,
status: "started"}` immediately — compiled pipelines run minutes to
days), `<name>_status` (polls a run by `workflow_id`), and for DBOS
`<name>_signal` (resumes a run parked at a HITL gate: `workflow_id` +
gate signal name + resume payload — so Claude can deliver approvals
itself). A DBOS run whose status stays `enqueued` means the emitted app
process isn't running against the registered system database.

## 4. Explain the reconnect caveat

A running `rote serve` picks up registry changes live — no restart of
the server, ever. But clients differ:

- **Claude Code** refreshes its tool list on the server's
  `list_changed` notification: newly registered pipelines appear
  immediately.
- **Claude Desktop and claude.ai** snapshot tools at connect time. A
  pipeline registered while they're connected appears only after a
  reconnect — restart Desktop or toggle the server off/on; on
  claude.ai, re-enable the connector.

Tell the user this proactively if they plan to use Desktop or claude.ai.
