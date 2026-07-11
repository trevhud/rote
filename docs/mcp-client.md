# The MCP client layer

How rote connects graduated workflows (and the eval harness) to
authenticated Streamable-HTTP MCP servers. This is the design record —
the user-facing surface is `rote mcp --help`.

## The problem

Graduated pipelines carry `mcp:` bindings (server / tool / transport)
and the DBOS adapter emits working FastMCP calls — but real remote MCP
servers authenticate with OAuth 2.1. Without a client that can run the
authorization flow, store tokens durably, and refresh them, MCP-backed
workflows only work against unauthenticated servers, and `eval --run`
can't measure skills whose tools need auth.

## Architecture

One durable credential layer, four consumers:

```
                 rote mcp add/login  (the human, once)
                        │
          ┌─────────────┴──────────────┐
          │ registry                   │ token store
          │ ~/.config/rote/mcp.json    │ ~/.local/share/rote/mcp-tokens/<name>.json
          │ names → urls, client creds │ tokens + client_info + expiry  (0600)
          └───────┬───────────┬────────┴───────────┬─────────────────┐
                  │           │                    │                 │
            rote CLI    emitted Python       eval --run        emitted TS
          (login/headers)  runtimes        (headersHelper →   (refresh-grant
                          (OAuth w/ store)  rote mcp headers)  client; planned)
```

- **Registry** (`rote mcp add <name> <url>`): logical server names —
  the same names pipelines carry in `mcp.server` — mapped to endpoints,
  optional pre-registered `client_id`/`client_secret` (for DCR-less
  servers: Slack/GitHub-class), optional scopes, optional static
  headers (API-key schemes). Written `0600`; overridable via
  `ROTE_MCP_CONFIG`.
- **Token store**: one JSON file per server name, `0600`, holding the
  SDK-shaped `tokens`, the registered `client_info` (so dynamic client
  registration never repeats), an absolute `expires_at`, and the
  discovered `token_endpoint`. Overridable via `ROTE_MCP_TOKEN_DIR`.
- **OAuth engine**: fastmcp's `OAuth` (a subclass of the official MCP
  SDK's `OAuthClientProvider`) — protected-resource discovery, PKCE,
  DCR/CIMD, refresh. rote plugs in durable storage via
  `ServerTokenKV`, an `AsyncKeyValue` shim mapping the provider's three
  storage collections onto the contract file's fields.

## URL resolution — the one rule

Everywhere (CLI, eval, emitted code):

1. explicit `url` on the IR binding (pipeline-pinned),
2. registry entry for the logical server name,
3. `ROTE_MCP_<SERVER>_URL` environment variable.

Exception: *at workflow runtime* the env var outranks the IR-recorded
URL (deployment context beats graduation-time capture) — see
`rote/mcp/_runtime_helper.py:resolve_url`.

## The token-file contract (version 1)

The store layout is a **cross-language contract**: Python writes it,
Python and (planned) TypeScript runtimes read and refresh through it.

```json
{
  "version": 1,
  "server_url": "https://mcp.example.com/mcp",
  "tokens": {
    "access_token": "…", "token_type": "Bearer",
    "expires_in": 3600, "refresh_token": "…", "scope": "…"
  },
  "expires_at": 1770000000.0,
  "client_info": { "client_id": "…", "…": "…" },
  "token_endpoint": "https://auth.example.com/token"
}
```

Rules: atomic replace, mode `0600`; `expires_at` is absolute (never
trust a stale relative `expires_in`); a reader treats a missing
`expires_at` as long-lived; a stale token with a `refresh_token` is
refreshable, without one it's dead (re-login). Contract parity between
the CLI store and the emitted helper is enforced by
`tests/test_mcp_client.py::test_runtime_helper_kv_is_byte_compatible_with_the_cli_store`.

## Emitted runtimes

The DBOS adapter ships `extracted/_rote_mcp.py` with every MCP-backed
app — the *verbatim source* of `rote.mcp._runtime_helper` (one tested
implementation, no drift; enforced by test). The emitted step calls
`call_mcp_tool(<server>, <ir-url>, <tool>, arguments)`: OAuth with the
shared store when the user has run `rote mcp login`, static registry
headers when configured, plain client otherwise. Emitted apps never
import rote.

## Park-on-auth (DBOS Python + DBOS TypeScript)

OAuth is interactive; durable workflows run unattended. The resolution:
auth is a *preflight* owned by the CLI (where a human is present), and
the emitted workflow only ever consumes cached tokens. When that fails
— expired token with no refresh path, a 401 from the server, or an
OAuth flow demanding authorization (the helper's OAuth subclass never
opens a browser) — the step raises `RoteMcpAuthNeeded` and the workflow
**parks durably instead of failing**:

1. The step's `should_retry` predicate exempts `RoteMcpAuthNeeded` from
   the retry budget — no number of retries produces a credential.
2. The workflow advertises what it's blocked on via the
   `rote_auth_status` workflow event (`{"awaiting": "<server>"}`).
   DBOS has no distinct WAITING status — a parked workflow is just
   PENDING, so it must say what it's waiting for.
3. It blocks on `DBOS.recv(topic="rote:auth:<server>")` (30-day
   timeout; the `:` in the prefix makes IR-signal collisions
   impossible — IR signal charsets exclude it).
4. `rote mcp login <server>`, after a successful dance, scans the app
   registry (`~/.local/share/rote/apps.json`, written by
   `rote emit`/`rote graduate`; `ROTE_APPS_PATH` overrides), finds
   PENDING workflows awaiting that server via `DBOSClient.get_event`,
   and sends each the release message. DBOS notifications persist
   per-topic, so releasing a workflow that hasn't quite reached its
   `recv` is race-free.
5. The workflow retries the step with the fresh credential and
   continues.

One subtlety: in a parallel wave, a sibling step's *stale* auth failure
can surface after the login already released (and was consumed by) an
earlier park. The emitted wrappers therefore retry the step once
immediately — it re-reads the token store — and only park if the fresh
attempt still needs auth. Proven end-to-end (real DBOS runtime, real
FastMCP server, cross-process release) in `tests/test_mcp_park_e2e.py`.

### The DBOS-TS park is cross-language, and serialization is the crux

The TypeScript emission mirrors the Python one (`RoteMcpAuthNeeded`
from the emitted `_roteMcp.ts` helper, a `shouldRetry` opt-out,
`DBOS.recv` on the same `rote:auth:<server>` topic, the same
`rote_auth_status` event) — and the *same Python release path* serves
both runtimes, because the two DBOS SDKs share one system-DB schema.
Two rules make the interop work; break either and the park silently
stops being discoverable or releasable:

1. **The emitted TS `setEvent` passes
   `{ serializationType: "portable" }`.** The TS default (superjson)
   raises `TypeError: Serialization js_superjson is not available` in
   the Python client; portable JSON is DBOS's documented cross-language
   format, dispatched per-row so the reader needs no config.
2. **The Python release `send` passes
   `serialization_type=WorkflowSerializationFormat.PORTABLE`.** The
   Python default (pickle) is opaque to a TS `DBOS.recv`. Portable is
   readable by both SDKs, so one code path releases Python and TS apps
   alike.

Two adjacent traps: detect the auth error in TS by **`name` string,
never `instanceof`** (DBOS reconstructs serialized errors as plain
`Error`s — class identity is lost; `name` and own enumerable fields
survive), and pass `load_input=False, load_output=False` when a Python
client lists a TS app's workflows (their inputs/outputs are superjson).
The full cross-language loop — TS parks, Python reads the event,
Python releases, TS resumes against a live MCP server — is proven on a
real Docker Postgres in `tests/test_mcp_park_ts_e2e.py`.

`rote mcp release <server>` triggers the same scan-and-release without
a login, for credentials fixed out-of-band (re-provisioned Worker
secrets, a synced token file, a registry entry switched to static
headers).

The remaining runtimes (Inngest, Cloudflare) still fail loud on dead
credentials; their parks are the known follow-up — Inngest via
`waitForEvent` + broadcast event release, Cloudflare via
`waitForEvent` + instance-addressed `sendEvent`.

## eval --run

`mcp_servers_for_pipeline` builds the `claude -p --mcp-config` and now
injects auth: static registry headers verbatim, or — for logged-in
servers — a `headersHelper` invoking
`<python> -m rote mcp headers <name>`. Claude Code re-runs the helper
on every connection and once more on a 401, so tokens refresh mid-run
on long trials (measured browser-automation runs park an agent for
90+ minutes; a static header would expire under it).

Two verified Claude Code behaviors this leans on (v2.1.205+):
`--strict-mcp-config` filters the server *list* but does not disable
header helpers, and headless print mode cannot run an OAuth flow itself
— pre-authentication via headers is the supported path.

## Headless login

`rote mcp login <name> --no-browser` prints the authorization URL
instead of opening a browser. The callback still lands on
`localhost:<port>` (spec-standard loopback redirect) — on SSH, forward
the port (`-L`). There is no fully non-interactive user-auth flow in
the MCP spec today (device grant is an open proposal); machine
identities (client-credentials) can be added later via the SDK's
`extensions` providers if a server supports them.

## TypeScript runtimes

**Node targets (DBOS-TS, Inngest) — shipped.** MCP-bound nodes emit a
working module calling `src/extracted/_roteMcp.ts` (source of truth:
`rote.adapters._ts_common.ROTE_MCP_HELPER_TS`): it reads the same
registry and token files the CLI writes, refreshes stale access tokens
with a plain `grant_type=refresh_token` POST to the stored
`token_endpoint` (deliberately *not* the SDK's `authProvider`, which
insists on running discovery and can silently fall through to a new
authorization flow), writes rotated refresh tokens back atomically, and
calls the tool via `@modelcontextprotocol/sdk` (^1.29 — the supported
v1 line; v2 splits the packages) with a retry-once-on-401. Proven by
`tests/test_mcp_ts_e2e.py`: Python logs in, compiled TS authenticates,
refreshes a forced-stale token, and rotates credentials Python reads
back.

**Cloudflare Workers — shipped.** No filesystem, no subprocess, so
credentials are *provisioned*: `rote mcp export <server>` turns a
completed login into Worker secrets (`ROTE_MCP_<S>_REFRESH_TOKEN` /
`_CLIENT_ID` / `_CLIENT_SECRET?` / `_TOKEN_ENDPOINT` / `_URL`;
`--json` pipes to `npx wrangler secret bulk`, the default dotenv form
pastes into `.dev.vars`). The emitted Workers helper
(`ROTE_MCP_WORKERS_HELPER_TS`) mints access tokens at runtime via the
refresh grant and caches them — with rotated refresh tokens — in the
`ROTE_MCP_TOKENS` KV namespace the emitted `wrangler.jsonc` declares
(per-isolate memory fallback when KV is absent). MCP results are
declared `Promise<never>` in emitted modules for the same
`Rpc.Serializable` reason stubs are. Static verification: the emitted
output typechecks against `@cloudflare/workers-types` v5 + the MCP SDK
(`test_cloudflare_mcp_output_typechecks`); a live workerd run of the
SDK's streamable-HTTP client remains a known follow-up.

## Security posture

File-backed store (`0600`, atomic writes) was chosen over OS keyrings
for v1: identical behavior on laptops, SSH boxes, and CI, and emitted
background workflows can read it without keychain prompts. Keyring
support is a planned opt-in. The registry may hold `client_secret`
(also `0600`). Nothing token-like is ever written into emitted code,
pipeline YAML, or logs; `scripts/sanity-check.sh` remains the gate.
