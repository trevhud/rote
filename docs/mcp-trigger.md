# Triggering compiled pipelines from Claude — `rote serve`

`rote serve` is a single MCP server that exposes every compiled
pipeline as a callable MCP tool. Compile a skill once, register it,
and any MCP client (Claude Code, Claude Desktop, claude.ai custom
connectors) can trigger the deterministic workflow instead of
re-running the fuzzy skill.

Built on FastMCP 4.x (>= 4.1.3, on MCP Python SDK v2). One server
answers both protocol eras: sessionless 2026-07-28 clients and
handshake-era (2025-11-25 and earlier) clients. The base install carries
only fastmcp's client half (`fastmcp-slim[client]`); the server half is
the serve extra:

```sh
pip install 'rote-cli[serve]'
```

## The flow

```
rote compile  →  deploy the runtime  →  rote register  →  rote serve  →  call from Claude
```

### 1. Compile and deploy

```sh
rote compile ./my-skill --runtime dbos --out ./my-skill-out
# → my-skill-out/compiled/pipeline.yaml
# → my-skill-out/runtime/dbos/{main.py,dbos-config.yaml,...}
```

Get the runtime side running: for DBOS, keep the emitted app alive in
worker mode (`python main.py --serve`, or `dbos start` — same thing);
for Temporal, run the emitted worker against your cluster; for
Cloudflare, `wrangler deploy`. `rote serve` triggers deployed
workflows; it does not host them.

### 2. Register

```sh
# DBOS (the default). The system database URL comes from
# --system-database-url, else $DBOS_SYSTEM_DATABASE_URL, else the
# emitted app's default SQLite file (found via runtime/dbos/main.py).
rote register ./my-skill-out

# Temporal (defaults: localhost:7233, namespace "default",
# task queue = pipeline.name, workflow type = the versioned name the
# Temporal adapter emitted for exactly this pipeline.yaml)
rote register ./my-skill-out --runtime temporal

# Cloudflare
rote register ./my-skill-out \
  --runtime cloudflare \
  --url https://my-pipeline.example.workers.dev
```

This upserts an entry in `~/.local/share/rote/serve-registry.json` (override with
`--registry` or `ROTE_SERVE_REGISTRY_PATH`): tool name from `pipeline.name`, description from
`pipeline.description`, `inputSchema` from the pipeline's input
contract, plus the trigger config. Registering the same pipeline again
updates the entry in place.

Useful flags: `--name` (override the tool name), `--system-database-url`
and `--queue-name` (DBOS), `--task-queue`, `--workflow-name`,
`--temporal-address`, `--temporal-namespace`, `--status-url`
(Cloudflare status route template containing `{workflow_id}`, if your
worker has one — the emitted default doesn't).

Note the DBOS/Temporal `--workflow-name` default is derived from the
pipeline *content hash* (matching the emitted `@DBOS.workflow` /
`@workflow.defn` name). If you re-compile and re-emit a changed
pipeline, re-register so the name stays in sync.

### 3. Serve

```sh
rote serve                      # stdio (what claude mcp add expects)
rote serve --http --port 8734   # Streamable HTTP at http://127.0.0.1:8734/mcp/
```

Each registry entry becomes two tools (three for DBOS):

- **`<name>`** — triggers the workflow. Returns immediately with
  `{workflow_id, status: "started", runtime}`. It never blocks on the
  workflow: compiled pipelines run minutes to days (HITL gates), and
  their durability lives in DBOS / Temporal / Cloudflare, not in this
  process.
- **`<name>_status`** — polls a run by `workflow_id`. For DBOS this
  reads the run's row from the system database (statuses `enqueued`,
  `pending`, `success`, `error`, `cancelled`, …); for Temporal it
  describes the workflow execution; for Cloudflare it needs
  `--status-url` (otherwise it points you at
  `wrangler workflows instances describe`).
- **`<name>_signal`** (DBOS only) — resumes a run parked at a HITL
  gate: `workflow_id` + the gate's `signal` name + a resume `payload`,
  delivered via `DBOSClient.send` on the gate's topic. The gate signals
  captured at register time become an enum on the `signal` parameter,
  and any other name is rejected (a message on an unknown topic would
  be silently swallowed by DBOS). This closes the loop: Claude can
  trigger a pipeline, watch it park at a gate (`status: "pending"`),
  and deliver the approval itself.

#### Built-in tools

Two tools are always present, registry or not, so a client can find
out what the server offers before anything is registered (and so MCP
directories such as Glama, which score a server by its tool
definitions, see tools on a fresh install):

- **`rote_list_pipelines`** lists every registered pipeline: name,
  description, runtime, a one-line trigger summary, the names of its
  trigger / status / signal tools, the `pipeline.yaml` path and the
  registration time, plus the registry path and whether the file
  exists. Optional `runtime` argument (`temporal`, `cloudflare`,
  `dbos`) filters the list. An empty or missing registry returns an
  empty `pipelines` array, not an error.
- **`rote_describe_pipeline`** takes a pipeline `name` and returns its
  full registry entry: the `input_schema` the trigger tool accepts, the
  trigger configuration with any URL password replaced by `***`, and
  the tool names. An unknown name fails with the list of registered
  names.

Both are read-only (annotated `readOnlyHint` / `idempotentHint`),
return structured content with an `outputSchema`, and re-read the
registry on every call, so they see a `rote register` immediately.

The `rote_` prefix is reserved for built-ins. `rote register` refuses
a tool name that starts with it (pick another with `--name`). If such
an entry reaches a hand-edited registry anyway, the built-in keeps its
name and the entry is left out of the tool list and of the built-ins'
output, with one warning on stderr naming the fix. The same rule
applies to companion names: an entry is skipped if its `<name>_status`
or `<name>_signal` would collide with a built-in.

Backends connect lazily: the server starts and lists tools even when
the runtime is down; unreachability surfaces as a clear tool-call error
(e.g. `Temporal at localhost:7233 ... is unreachable`). Triggering DBOS
or Temporal pipelines additionally needs that runtime's extra in the serve
environment (`pip install 'rote-cli[serve,dbos]'`, or `[serve,temporal]`).

#### The DBOS operational contract

DBOS has no orchestrator or HTTP endpoint — the trigger contract is the
**system database** shared between `rote serve` and the emitted app:

1. Triggering is a database write. `DBOSClient.enqueue` inserts an
   `ENQUEUED` workflow row; it succeeds even when no app is running.
2. For the run to *execute*, the emitted app process must be running
   against the same system database — `python main.py --serve` or
   `dbos start` — because the app process is what dequeues and executes.
   The registered `workflow_name` (PascalCase name + pipeline hash) and
   `queue_name` (`<pipeline.name>-queue`) must match what that app's
   code registers, which is exactly what `rote register --runtime dbos`
   derives.
3. A run stuck in status `enqueued` means no matching app process has
   picked it up yet: the app isn't running, points at a different
   system database, or (after a re-emit without re-registering) the
   workflow name hash no longer matches.
4. HITL resume (`<name>_signal` → `DBOSClient.send`) is also just a
   durable database write; the parked run resumes whether it parked
   yesterday or the app restarted since.
5. SQLite system databases work cross-process and are what the emitted
   app defaults to, but DBOS supports them for development and testing
   only — use Postgres in production (`DBOS_SYSTEM_DATABASE_URL` on
   both the app and `rote register`).

On MCP Tasks: FastMCP 4 ships them as the optional
`io.modelcontextprotocol/tasks` extension (`fastmcp-tasks`), but running
triggers as tasks would tie multi-day workflow observability to the
server process's lifetime. The `_status` polling pattern is deliberate.

Idle sessions: since fastmcp 4.1 a handshake-era Streamable HTTP
session expires after 30 idle minutes. Nothing about a run lives in the
session (the `workflow_id` is the handle), so a client that reconnects
polls the same id. stdio has no session timeout.

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
2. **Push:** a file watcher tells every connected client when the
   registry changes. FastMCP doesn't emit this for provider-sourced
   changes on its own, and the two protocol eras need different
   channels, so the server does both:
   - handshake-era connections get `notifications/tools/list_changed`
     on their standalone stream (the server tracks each connection);
   - sessionless 2026-07-28 clients have no session to push to. They
     open a `subscriptions/listen` stream (SEP-2575) and the change
     arrives on it. FastMCP 4.1 registers no listen handler, so `rote
     serve` installs the SDK's; without it the server would advertise
     `tools.listChanged: false` on that era.

The caveat: push only helps clients that honor it. **Claude Code
refreshes its tool list on `list_changed`. Claude Desktop and
claude.ai currently do not** — they snapshot tools at connect time, so
a pipeline registered while they're connected appears only after a
reconnect (Desktop: restart or toggle the server; claude.ai: re-enable
the connector). No restart of `rote serve` is ever needed.

## The registry file

`~/.local/share/rote/serve-registry.json`, schema version 1. One entry per tool:

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
