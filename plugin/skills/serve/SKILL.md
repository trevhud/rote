---
name: serve
description: >-
  Wire a graduated rote pipeline up as an MCP tool so Claude can trigger the
  deployed workflow directly. Use when the user says "register my graduated
  pipeline", "serve my pipelines over MCP", "trigger the workflow from
  Claude", "hook the pipeline up to Claude", or asks what to do after
  `rote graduate` and deployment. Covers `rote register` and `rote serve`
  plus the `claude mcp add` wiring.
---

# Serve graduated pipelines as MCP tools

`rote serve` is one MCP server exposing every registered pipeline as a
callable tool. The flow:

```
rote graduate → deploy the runtime → rote register → rote serve → call from Claude
```

`rote serve` **triggers deployed workflows; it does not host them.**
MCP triggering supports the `temporal` and `cloudflare` runtimes today
(not `dbos`).

The CLI ships on PyPI as `rote-cli` with an executable named `rote`,
so every invocation is `uvx --from rote-cli rote <args>` (never
`uvx rote-cli ...`). For unreleased features, substitute the source:
`uvx --from git+https://github.com/trevhud/rote rote <args>`.

## 1. Check preconditions

- A graduate output directory exists (contains `graduated/pipeline.yaml`).
- The runtime side is deployed: a worker running against the user's
  Temporal cluster, or `wrangler deploy` done for Cloudflare. If not,
  stop and help with that first.

## 2. Register the pipeline

```sh
# Temporal (defaults: localhost:7233, namespace "default",
# task queue = pipeline.name, workflow type = the emitted versioned name)
uvx --from rote-cli rote register <out-dir>

# Cloudflare
uvx --from rote-cli rote register <out-dir> --runtime cloudflare \
  --url https://<worker>.workers.dev
```

This upserts `~/.rote/registry.json`. Re-registering updates in place.
**After re-graduating a changed skill, register again** — the Temporal
workflow type name is derived from the pipeline content hash and must
stay in sync with the emitted code.

## 3. Add the MCP server to Claude

`rote serve` needs the `serve` extra (FastMCP), so the spec includes it:

```sh
claude mcp add --scope user rote -- uvx --from 'rote-cli[serve]' rote serve
```

For unreleased features, use the GitHub source instead:

```sh
claude mcp add --scope user rote -- \
  uvx --from 'rote-cli[serve] @ git+https://github.com/trevhud/rote' rote serve
```

Verify with `claude mcp list`. Each registry entry becomes two tools:
`<name>` (starts a run, returns `{workflow_id, status: "started"}`
immediately — graduated pipelines run minutes to days) and
`<name>_status` (polls a run by `workflow_id`).

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
