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
implementation, no drift; enforced by test). The emitted step opens
`mcp_client(<server>, <ir-url>)`: OAuth with the shared store when the
user has run `rote mcp login`, static registry headers when configured,
plain client otherwise. Emitted apps never import rote.

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

## TypeScript runtimes (planned, PR2/PR3)

- **Node targets (DBOS-TS, Inngest)**: an emitted refresh-grant client
  — read the contract file, POST `grant_type=refresh_token` to the
  stored `token_endpoint` when stale, write rotated tokens back
  atomically. No SDK dependency; ~60 lines of fetch.
- **Cloudflare Workers** (no filesystem, no subprocess): provision
  refresh credentials as Worker secrets (`rote mcp export`), refresh at
  runtime, persist rotated refresh tokens to a KV binding added to the
  emitted `wrangler.jsonc`.

## Security posture

File-backed store (`0600`, atomic writes) was chosen over OS keyrings
for v1: identical behavior on laptops, SSH boxes, and CI, and emitted
background workflows can read it without keychain prompts. Keyring
support is a planned opt-in. The registry may hold `client_secret`
(also `0600`). Nothing token-like is ever written into emitted code,
pipeline YAML, or logs; `scripts/sanity-check.sh` remains the gate.
