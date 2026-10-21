# The MCP client layer

How rote connects compiled workflows (and the eval harness) to
authenticated Streamable-HTTP MCP servers. This is the design record —
the user-facing surface is `rote mcp --help`.

## The problem

Compiled pipelines carry `mcp:` bindings (server / tool / transport)
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
                          (OAuth w/ store)  rote mcp headers)  client, shipped)
```

- **Registry** (`rote mcp add <name> <url>`): logical server names —
  the same names pipelines carry in `mcp.server` — mapped to endpoints,
  optional pre-registered `client_id`/`client_secret` (see
  [Client registration](#client-registration)), optional scopes,
  optional static headers (API-key schemes). Written `0600`;
  overridable via `ROTE_MCP_CONFIG`. Unknown fields are kept, because
  every rote version on the machine shares the file.
- **Learned auth** (`mcp-auth-learned.json` beside the registry,
  `rote.mcp.learned`): whether each server needs a login at all
  (`oauth` or `none`), keyed by name and normalized URL so a repoint
  invalidates it. `rote mcp login` records it (a server whose
  `tools/list` succeeds with no challenge is `none`), `rote mcp add
  --auth` sets it, and a token for the endpoint or an observed auth park
  overrides a learned `none`. Nothing learned reads as "auth unknown",
  and `rote emit` only says a run "will park" for a server known to
  need OAuth. It is a separate file because rote 0.14.0 and older reject
  unknown registry fields.
- **Token store**: one JSON file per server name, `0600`, holding the
  SDK-shaped `tokens`, the `client_info` the login used (so registration
  never repeats), an absolute `expires_at`, and the `token_endpoint` and
  `issuer` of the authorization server. Overridable via
  `ROTE_MCP_TOKEN_DIR`.
- **OAuth engine**: fastmcp 4's `OAuth` (a subclass of the official MCP
  SDK v2's `OAuthClientProvider`): protected-resource discovery, PKCE,
  RFC 9207 `iss` validation, issuer-bound credentials, CIMD, DCR,
  refresh. rote plugs in durable storage via `ServerTokenKV`, an
  `AsyncKeyValue` shim mapping the provider's three storage collections
  onto the contract file's fields, and the registration rules below.

## Protocol eras

MCP 2026-07-28 removed the `initialize` handshake and sessions. Every
client rote ships prefers it and falls back to the 2025 handshake:

| Client | Library | Negotiation |
| --- | --- | --- |
| `rote mcp login`, live compiler tools, the emitted Python helper | fastmcp 4 / MCP SDK v2 | `auto` (fastmcp's default): probe `server/discover`, fall back |
| emitted `_roteMcp.ts` (DBOS-TS, Inngest) and the Workers helper | `@modelcontextprotocol/client` 2.x | `versionNegotiation: { mode: "auto" }`, passed explicitly |

The explicit `auto` matters: the v2 TS client defaults to `legacy`,
which behaves exactly like v1 against a modern-only server
(`-32022 Unsupported protocol version`). v1
(`@modelcontextprotocol/sdk` 1.x) has no 2026-07-28 support at all,
which is why the emitted helpers moved to v2.
`tests/test_mcp_era_matrix_e2e.py` runs both helpers against four real
TypeScript SDK servers (legacy stateless, legacy sessionful, dual-era,
modern-only) and asserts the era each server saw, so a client that
silently stays on the old era against a dual-era server fails. Its
negative half proves the modern-only fixture really refuses a
`legacy`-mode client.

## Client registration

`rote mcp login` follows the spec's priority order:

1. **A pre-registered client** from the registry (`rote mcp add <name>
   <url> --client-id <id> [--client-secret <secret>]`). Always wins,
   even when the server offers more.
2. **rote's Client ID Metadata Document (CIMD)** when the authorization
   server advertises `client_id_metadata_document_supported`. The
   client_id is the URL `https://roteskills.com/oauth/rote-cli.json`;
   the document's source is [`docs/oauth/rote-cli.json`](oauth/rote-cli.json).
3. **Dynamic client registration (DCR)** when the server advertises a
   `registration_endpoint`. Deprecated in 2026-07-28, still the only
   option for most hosted servers today.
4. **Otherwise, an error** naming the fix: register an OAuth app with
   that authorization server and add it with `--client-id`. The SDK on
   its own would POST to a guessed `<resource-origin>/register`, which
   for GitHub is not GitHub's authorization server at all; rote refuses
   to send that request.

`ROTE_MCP_CLIENT_METADATA_URL` overrides the CIMD URL (the e2e suite
points it at a local TLS copy of the document), and an empty value
turns CIMD off.

The SDK does not fall back to DCR when a CIMD server rejects the
document, so rote checks it first. When the authorization server's
metadata advertises CIMD, the login fetches the document URL once per
process and offers it only if it returns 200 and equals the packaged
copy with `client_id` set to the URL. Otherwise it prints one line and
registers with DCR. Servers that do not advertise CIMD never trigger
the fetch. The fetch has a 5 second total deadline, not just httpx's
per-read timeout, so a host that answers one byte at a time cannot hold
the login open.

### The pre-registered client path (GitHub-class servers)

Servers whose authorization server offers neither CIMD nor DCR (GitHub,
Asana, HubSpot in the July 2026 probe) need an OAuth app registered by
hand. Register one with the provider, set its redirect URI to
`http://localhost:33418/callback` (add 33419 to 33421 too if the
provider accepts several, so a busy port does not block the login),
then:

```sh
rote mcp add github https://api.githubcopilot.com/mcp/ \
  --client-id <id> --client-secret <secret>
rote mcp login github
```

A pre-registered client is bound to an issuer from the start:
`rote mcp add --client-id` records the authorization server the
resource's protected-resource metadata names at that moment (or
`--issuer <url>`, which wins), and after a login the token file's
`issuer` is the binding. What gets recorded decides which server the
client secret is later offered to, so discovery accepts only a document
whose `resource` is this URL (RFC 9728 section 3.3; on a shared host the
origin-level document describes some other resource), and only HTTPS
for both the resource and the issuer, loopback excepted. `--issuer`
must be HTTPS too. Anything else records nothing and says why. If the server's protected-resource
metadata later names a different authorization server, the login stops
with an error before the browser opens instead of offering that client
to the new server, or quietly switching to CIMD or DCR there (spec:
clients SHOULD surface an error rather than use mismatched credentials).

### Loopback redirect ports

A CIMD authorization server validates the redirect URI against the
document by exact string comparison (CIMD draft section 4.2). Neither
the CIMD draft nor the MCP spec defines a loopback port wildcard, and
RFC 8252's any-port rule is per-server behavior. So the document lists
a fixed set, `localhost` and `127.0.0.1` on ports 33418 to 33421, and
the login binds the first free one (`rote.mcp.auth.CALLBACK_PORTS`; a
unit test pins the document to that constant). This works without the
server being lenient. `--callback-port` with any other port skips CIMD
for that login, since it would be rejected by exactly the servers that
support it. If all four ports are busy the login fails and says so.
Selecting a port binds it: the login holds the listening sockets from
selection until the callback server runs on them, so nothing can take
the port after the redirect URI names it, and every path (success,
failure, a still-valid token that needs no callback) releases them.

### `iss` and issuer binding

The SDK validates RFC 9207 `iss` before redeeming the code: a present
`iss` must equal the recorded issuer (whether or not the server
advertises support), and a missing one is rejected when the server
advertises `authorization_response_iss_parameter_supported`. rote
surfaces the failure as a login error and no token request is sent.
DCR credentials carry the issuer they were registered with, and a login
against a different authorization server re-registers instead of
presenting them there. The SDK reads that binding only from
`client_info.issuer` and treats a missing one as matching every
issuer, so both stores fill it in from the file's top-level `issuer`
when it is missing, login writes it to both places, and an interactive
login drops a registered client that has no issuer anywhere (a v1 file)
and registers again rather than offer it, secret included, to whatever
server the resource names next.

### A login belongs to one endpoint

Token files are keyed by server name, and a name can be pointed at a
new endpoint (`rote mcp add <name> <new-url>`, or
`ROTE_MCP_<SERVER>_URL` at run time). A file whose `server_url` is a
different endpoint is treated as absent, compared after normalizing
(scheme and host case, default port, trailing slash): `rote mcp
headers` refuses and names `rote mcp login`, `rote mcp list` shows the
server as not authenticated, a login starts fresh, and the emitted
Python and Node helpers raise `RoteMcpAuthNeeded` so the run parks.
The old endpoint's bearer token never reaches the new host. `tests/test_mcp_oauth_registration_e2e.py` runs
the real login against a local authorization server switchable between
CIMD-only, DCR-only, both, and neither, with correct, mismatched, and
missing `iss`, and two issuers to switch between. Every test asserts on
the server's hit log, because the failures that matter are requests
that should never be sent.

### Refresh goes where login found the token endpoint

Before any discovery, the SDK refreshes at `<resource-origin>/token`.
For an authorization server on another host or a path issuer that
404s, the provider falls into a full authorization, and an unattended
caller parks on a credential that was fine. The CLI and the emitted
Python helper send the refresh to the token file's `token_endpoint`
instead (`retarget_token_request`). The TS helpers always did.

## URL resolution

What every emitted step and agent loop does (`resolve_url` in
`rote/mcp/_runtime_helper.py`, and `resolveUrl` in the Node TS helper):

1. `ROTE_MCP_<SERVER>_URL` environment variable,
2. registry entry for the logical server name,
3. the `url` recorded on the IR binding at compilation time.

Deployment context beats compilation-time capture. Cloudflare Workers
have no registry: the `ROTE_MCP_<SERVER>_URL` var or secret, then the IR
`url`. Nothing resolving is a `RoteMcpConfigError`, which fails the step
without retries.

## The token-file contract (version 2)

The store layout is a **cross-language contract**: Python writes it,
Python and TypeScript runtimes read and refresh through it.

```json
{
  "version": 2,
  "server_url": "https://mcp.example.com/mcp",
  "tokens": {
    "access_token": "…", "token_type": "Bearer",
    "expires_in": 3600, "refresh_token": "…", "scope": "…"
  },
  "expires_at": 1770000000.0,
  "client_info": { "client_id": "…", "…": "…" },
  "token_endpoint": "https://auth.example.com/token",
  "issuer": "https://auth.example.com"
}
```

Version 2 added `issuer`, the authorization server the credentials
belong to (the spec keys credentials by issuer). Readers accept both
versions; a v1 file is a v2 file whose issuer is unknown, and the next
write stamps version 2.

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

## Park-on-auth (all four MCP-capable runtimes)

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
   `rote_auth_status` workflow event
   (`{"awaiting": "<server>", "node": "<node id>", "park": <n>}`). The
   number tells two consecutive parks apart when an observer polls
   slower than a release followed by a re-park.
   DBOS has no distinct WAITING status — a parked workflow is just
   PENDING, so it must say what it's waiting for.
3. It blocks on `DBOS.recv(topic="rote:auth:<server>")` (30-day
   timeout; the `:` in the prefix makes IR-signal collisions
   impossible — IR signal charsets exclude it).
4. `rote mcp login <server>`, after a successful dance, scans the app
   registry (`~/.local/share/rote/apps.json`, written by
   `rote emit`/`rote compile`; `ROTE_APPS_PATH` overrides), finds
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

### The Inngest park releases by broadcast, not discovery

Inngest has no parked-run discovery API and needs none: **events fan
out to every matching `waitForEvent` waiter**, so the release is one
POST per pipeline (`<pipeline>/rote.auth.<server>` — dots, not the
DBOS runtimes' `:` prefix, which has no documented status in Inngest
event names; gate signals can't contain dots, so collision is
impossible). `release_parked_workflows` sends it through the dev
server by default, Inngest Cloud when `INNGEST_EVENT_KEY` is set, or
wherever `ROTE_INNGEST_EVENT_URL` points.

The emitted park is a `runParkable` loop with three Inngest-specific
moves, all verified live (`tests/test_mcp_park_inngest_e2e.py`):

1. **Fresh step ids per attempt** (`<id>`, `<id>-retry-1`,
   `<id>-auth-wait-1`, …) — the executor memoizes by id, so a retry
   loop must never reuse one.
2. **`NonRetriableError` wrapping inside `step.run`** — Inngest's
   `retries` budget applies per step with managed exponential backoff;
   an auth failure would otherwise burn attempts for minutes before
   the catchable `StepError` surfaces. Detection at the body walks the
   error tree by `name` with a message-text fallback (serialization
   can prune deep cause chains).
3. **Retry-once-before-parking carries the race**: unlike DBOS and
   Cloudflare, Inngest does *not* buffer events for waits that haven't
   started — a release broadcast landing before the run reaches its
   `waitForEvent` is lost. The immediate retry re-reads the credential
   store the release just fixed, so nothing strands.

One deployment note: the Node helper reads the credential store on the
host *serving the app* — for a deployed Inngest service, the login (or
a token-file sync) must happen there; the broadcast then wakes the run
from anywhere.

### The Cloudflare park releases by blast, and KV supersedence enables it

Workflows has no broadcast — every send targets one instance — but
**events sent before an instance reaches its `waitForEvent` are
buffered per-instance** (documented), so `rote mcp release` doesn't
need to know which instances are parked: it sends `rote_auth_<server>`
to every non-terminal instance via the Cloudflare REST API
(`CLOUDFLARE_API_TOKEN` + `CLOUDFLARE_ACCOUNT_ID`). Parked instances
consume it; the rest buffer it harmlessly. Local dev has no REST
surface — `npx wrangler workflows instances send-event … --local` is
the manual channel there.

The emitted park loop's Cloudflare specifics, all verified live on
workerd (`tests/test_mcp_park_cf_e2e.py`):

1. **`NonRetryableError`** (from `cloudflare:workflows`) is the retry
   opt-out — Workflows has no should-retry predicate, and a dead
   credential would otherwise burn `retries.limit` attempts of delay.
2. **`waitForEvent` THROWS on timeout** (default 24h!) — the park sets
   an explicit `"30 days"`, and expiry failing the instance is the
   intended terminal behavior.
3. **The credential fix is provisioning, not login**: Workers read
   secrets, not the laptop token store — `rote mcp export` +
   `wrangler secret bulk`, then release. The `ROTE_MCP_TOKENS` KV
   cache deliberately supersedes provisioned secrets (rotation), which
   is also what makes an in-session fix testable: writing a fresh
   token into KV is the same read path the helper takes on retry.

The park e2e also closed a long-standing gap: the post-release retry
drives `callMcpTool` through the real `@modelcontextprotocol/client`
streamable-HTTP client **inside workerd** against a live server — the
Workers MCP output was previously only ever typechecked.

## Seeing a park from `rote run`

Unattended, a park is the right behavior. Under `rote run` it used to
look like a hang. The run-side watcher (`rote.mcp.parked`) closes that
gap on all four parking runtimes:

| Runtime | How the runner sees the park | How it releases this run |
| --- | --- | --- |
| dbos, dbos-ts | the `rote_auth_status` event (server + node) | `DBOSClient.send` on `rote:auth:<server>`, portable |
| inngest | a `rote-auth-park {json}` line in the app log | broadcast `<pipeline>/rote.auth.<server>` to the run's own dev server |
| cloudflare | the same line in the `wrangler dev` log | it does not (see below) |

The marker line exists because neither local dev server reports a
truthful "waiting on event X" status. Both runtimes replay workflow
code, so the line can print again after a release; its `wait` field
(the wait's own step id) de-duplicates it.

Each park prints once, naming the node, the server, and the command
that fixes it (read from the registry: `rote mcp login`, or `rote mcp
add` first for an unregistered server, or a header update for a
static-header one). Then:

- **On a terminal** (stdin and stderr both TTYs), the runner offers to
  run the browser login right there. Accepting runs the same OAuth dance
  as `rote mcp login` and releases the run. Time spent in the prompt and
  the browser extends `--timeout`.
- **Off a terminal** nothing prompts and no browser opens. The runner
  watches the server's token file and registry entry, and when either
  changes (a `rote mcp login` in another terminal, say) it releases its
  own run. A change counts only once the token file holds an access
  token: a login writes its client registration first, and releasing on
  that write re-parks the run. On DBOS it also waits a few seconds,
  because `rote mcp login` already releases registered DBOS apps itself.
  On Inngest the runner is the only releaser there is: the registry-wide
  release targets the default Inngest port, which a `rote run` dev
  server never uses.
- **Cloudflare is visibility only.** A Worker reads its own secrets, not
  the local token store, so a login on this machine changes nothing it
  can see. The runner prints the real fix (`rote mcp login`, then
  `rote mcp export <server> >> .dev.vars`, then re-run), offers no
  login, does not self-release, and keeps its HITL gate payloads for
  real gates (an auth park looks parked to the stability heuristic too).

Emitted code stays never-interactive; all of this is CLI-side.
`rote mcp parked [server]` lists parked DBOS workflows across the
registered apps without releasing them (Inngest and Cloudflare offer no
"which runs wait on which event" query, so they are reported as
skipped). Proven live in `tests/test_mcp_park_e2e.py`: a piped `rote
run` that resumes on a credential fix, and a `rote run` on a pty that
logs in against a real OAuth-protected server and completes.

## Read-only runs

`rote run` is read-only by default on both sides, so a skill run and its
pipeline can be compared side by side without either one writing:

- **One rule.** A tool may be called when the server marks it
  `readOnlyHint`, or when the user registered the whole server read-only
  with `rote mcp add <name> <url> --read-only` (stored as `read_only` in
  the registry). The declaration covers the registered URL only:
  `ROTE_MCP_<SERVER>_URL` wins over the registry, so a name registered
  read-only can be pointed at another endpoint, and that endpoint's tools
  are judged by their own annotations. Many real servers set no annotations at all (DeepWiki is
  one), so the declaration is the only way to read from them unattended.
  The predicate is `rote.mcp._runtime_helper.tool_is_read_only`, used by
  the skill side's tool allowlist (`rote.mcp.live_tools.read_only_tools`)
  and by every emitted helper.
- **Skill runs** get an allowlist of the tools that pass (unchanged).
- **Pipeline runs** get `ROTE_MCP_READ_ONLY=1` and
  `ROTE_MCP_READ_ONLY_SERVERS` (a JSON object of the declared servers and
  their registered URLs) in the app's
  environment, or as `wrangler dev --var` on Cloudflare, since a Worker
  cannot read the registry. The Python, Node and Workers helpers check
  each call against the `tools/list` they already fetch and refuse a
  tool that may write before calling it, as a `RoteMcpConfigError`: not
  retried, and printed as one line. An agent loop's MCP tools are not held
  to read-only calls yet, so a read-only run refuses to start one.
- `--allow-writes` lifts the gate on either side. An emitted app run
  directly, outside `rote run`, has no gate.

What the gate is not:

- **It covers MCP calls only.** A pipeline emitted with `--backend api`
  calls vendor HTTP APIs from its own code, and those requests, writes
  included, are not gated.
- **It is a guard for cooperative code, not a sandbox.** The emitted
  helpers check before they call; code that reaches a server another way
  (its own client, a subprocess, a hand edit to the helper) is not
  stopped, and neither is a server whose `readOnlyHint` is wrong.

The rote-cloud binding helper has no gate: a deployed run is not a
`rote run`.

## A server asking for input mid-call

A 2026-07-28 server can answer a tool call with `input_required`
(elicitation, sampling, or roots requests) and expect the client to
retry with the answers, or refuse the call with `-32021` because the
client declared no capability the tool needs. An emitted step has
nobody to answer, and no login answers it either, so it is not an auth
failure: every helper raises `RoteMcpInputRequired` naming the server,
the tool, and what was asked, and every runtime fails the step at once
(the retry opt-outs exempt it, and the park loops never see it). A
result that carries only `requestState`, asking for nothing, is still
retried as the server wants. Proven by
`tests/test_mcp_era_matrix_e2e.py` (both helpers, a real TS v2 server)
and `tests/test_mcp_park_e2e.py` (a real DBOS run fails in one call,
with a retry budget of 3, and never parks).

Parking on such a request and resuming with the persisted
`requestState` is a possible follow-up; it is not built.

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
calls the tool via `@modelcontextprotocol/client` (^2.3, negotiation
`auto`; see [Protocol eras](#protocol-eras)) with a retry-once-on-401.
A 401 arrives as `SdkHttpError` with `status === 401`, the version
probe's included. Proven by
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
output typechecks against `@cloudflare/workers-types` v5 + the MCP
client (`test_cloudflare_mcp_output_typechecks`); the park-on-auth e2e
(`tests/test_mcp_park_cf_e2e.py`) additionally drives the SDK's
streamable-HTTP client through a live workerd run against a real
server.

**Reading a tool's result, the same on every runtime.** Structured
content wins (fastmcp's `{"result": ...}` wrapper removed when marked).
Otherwise each text block is JSON-parsed and kept verbatim when it is not
JSON. One block gives its value, several give a list, and none gives
`null`. Image,
audio and embedded-resource blocks are skipped. `NaN`, `Infinity` and
`-Infinity` are not JSON, so both sides keep that text as a string.
Python refuses them through `parse_constant`, and TS `JSON.parse`
rejects them anyway. `tests/test_mcp_text_payload_parity_e2e.py` holds
the Python, Node and Workers helpers to one answer against a real
server. One difference is left on purpose: **integers beyond 2^53 lose
precision on TS.** `JSON.parse` reads every number as an IEEE double, so
`12345678901234567890` comes back as `12345678901234567000`, where Python
keeps it exact. A tool that returns large IDs as JSON numbers should
return them as strings if a TS runtime will read them.

## Security posture

File-backed store (`0600`, atomic writes) was chosen over OS keyrings
for v1: identical behavior on laptops, SSH boxes, and CI, and emitted
background workflows can read it without keychain prompts. Keyring
support is a planned opt-in. The registry may hold `client_secret`
(also `0600`). Nothing token-like is ever written into emitted code,
pipeline YAML, or logs; `scripts/sanity-check.sh` remains the gate.
