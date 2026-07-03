# Triggering graduated pipelines from Claude — `rote serve`

`rote serve` is a single MCP server that exposes every graduated
pipeline as a callable MCP tool. Graduate a skill once, register it,
and any MCP client (Claude Code, Claude Desktop, claude.ai custom
connectors) can trigger the deterministic workflow instead of
re-running the fuzzy skill.

Built on FastMCP 3.x (>= 3.4.2). Install with the serve extra:

```sh
pip install 'rote[serve]'
```

## The flow

```
rote graduate  →  deploy the runtime  →  rote register  →  rote serve  →  call from Claude
```

### 1. Graduate and deploy

```sh
rote graduate ./my-skill --runtime temporal --out ./my-skill-out
# → my-skill-out/graduated/pipeline.yaml
# → my-skill-out/runtime/temporal/{workflow.py,activities.py}
```

Run the emitted worker against your Temporal cluster (or `wrangler
deploy` the Cloudflare output). `rote serve` triggers deployed
workflows; it does not host them.

### 2. Register

```sh
# Temporal (defaults: localhost:7233, namespace "default",
# task queue = pipeline.name, workflow type = the versioned name the
# Temporal adapter emitted for exactly this pipeline.yaml)
rote register ./my-skill-out

# Cloudflare
rote register ./my-skill-out \
  --runtime cloudflare \
  --url https://my-pipeline.example.workers.dev
```

This upserts an entry in `~/.rote/registry.json` (override with
`--registry`): tool name from `pipeline.name`, description from
`pipeline.description`, `inputSchema` from the pipeline's input
contract, plus the trigger config. Registering the same pipeline again
updates the entry in place.

Useful flags: `--name` (override the tool name), `--task-queue`,
`--workflow-name`, `--temporal-address`, `--temporal-namespace`,
`--status-url` (Cloudflare status route template containing
`{workflow_id}`, if your worker has one — the emitted default doesn't).

Note the Temporal `--workflow-name` default is derived from the
pipeline *content hash* (matching the emitted `@workflow.defn` name).
If you re-graduate and re-emit a changed pipeline, re-register so the
name stays in sync.

### 3. Serve

```sh
rote serve                      # stdio (what claude mcp add expects)
rote serve --http --port 8734   # Streamable HTTP at http://127.0.0.1:8734/mcp/
```

Each registry entry becomes two tools:

- **`<name>`** — triggers the workflow. Returns immediately with
  `{workflow_id, status: "started", runtime}`. It never blocks on the
  workflow: graduated pipelines run minutes to days (HITL gates), and
  their durability lives in Temporal / Cloudflare, not in this process.
- **`<name>_status`** — polls a run by `workflow_id`. For Temporal this
  describes the workflow execution; for Cloudflare it needs
  `--status-url` (otherwise it points you at
  `wrangler workflows instances describe`).

Backends connect lazily: the server starts and lists tools even when
the runtime is down; unreachability surfaces as a clear tool-call error
(e.g. `Temporal at localhost:7233 ... is unreachable`).

On MCP Tasks (SEP-1686): FastMCP 3.4 ships server-side support, but the
extension is a spec RC and running triggers as tasks would tie
multi-day workflow observability to the server process's lifetime. The
`_status` polling pattern is deliberate; revisit once the 2026-07-28
spec lands and Claude clients request task augmentation.

### 4. Call from Claude

```sh
claude mcp add --scope user rote -- rote serve
```

(or `rote serve --registry /path/to/registry.json` after the `--` to
pin a non-default registry). Then, in any Claude Code session:

> "Kick off the bdr-campaign pipeline for ExampleDrug / example
> condition with a quota of 25, then check on it."

Claude calls `bdr-campaign` with the typed input (the `inputSchema` is
the pipeline's input contract, so required fields are enforced), gets a
`workflow_id`, and polls `bdr-campaign_status`.

For the HTTP transport:

```sh
claude mcp add --scope user --transport http rote http://127.0.0.1:8734/mcp/
```

## Live registration and the reconnect caveat

A running `rote serve` notices registry changes without a restart, two
ways:

1. **Pull:** the tool list is re-read from the registry file on every
   `tools/list`, so any client that re-lists sees new pipelines
   immediately.
2. **Push:** a file watcher sends `notifications/tools/list_changed`
   to every connected session when the registry changes. (FastMCP 3.4
   doesn't emit this for provider-sourced changes on its own; the
   server tracks sessions and uses the SDK's
   `send_tool_list_changed()` directly.)

The caveat: push only helps clients that honor it. **Claude Code
refreshes its tool list on `list_changed`. Claude Desktop and
claude.ai currently do not** — they snapshot tools at connect time, so
a pipeline registered while they're connected appears only after a
reconnect (Desktop: restart or toggle the server; claude.ai: re-enable
the connector). No restart of `rote serve` is ever needed.

## The registry file

`~/.rote/registry.json`, schema version 1. One entry per tool:

```json
{
  "version": 1,
  "entries": [
    {
      "name": "bdr-campaign",
      "description": "End-to-end BDR outreach campaign workflow ...",
      "pipeline_yaml": "/abs/path/to/pipeline.yaml",
      "input_schema": { "type": "object", "properties": { "...": {} } },
      "trigger": {
        "runtime": "temporal",
        "address": "localhost:7233",
        "namespace": "default",
        "task_queue": "bdr-campaign",
        "workflow_name": "BdrCampaign_ac861059"
      },
      "registered_at": "2026-07-02T00:00:00+00:00"
    }
  ]
}
```

`input_schema` is stored denormalized (not re-derived at serve time) so
the served contract is exactly what you registered. Today it's
synthesized permissively from the IR's untyped `required`/`optional`
name lists; when the IR grows a structured `input_schema` on
`PipelineInput`, `rote register` will prefer it automatically.

The file is the whole contract between `rote register` and
`rote serve` — same philosophy as the driver layer's
`work_dir/pipeline.yaml`. Edit it by hand if you like; writes from
`rote register` are atomic (tmp file + rename).
