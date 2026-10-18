# Probes for the MCP 2026-07-28 design doc

The scripts behind the appendix of [`../mcp-2026-07-28.md`](../mcp-2026-07-28.md).
They check the ecosystem, not rote, so they are **not tests**:

- they hit third-party servers rote does not control;
- the results drift as those servers change.

Rerun them before acting on the doc, or before a release that touches MCP.

None of them authenticate, and none need credentials. The login-path probe
runs rote's real OAuth code but aborts before any request with side effects:

- it blocks the DCR `POST` before it is sent;
- it intercepts the authorization redirect and never opens it.

No inference anywhere.

## Environments

Run from the repo root. Python scripts run with `-I` so nothing from the
current directory is imported.

```sh
P=docs/design/mcp-2026-07-28-probes

# main's stack: fastmcp 3.x + mcp 1.x (+ dbos for A7)
uv venv /tmp/rote-probe-main
uv pip install --python /tmp/rote-probe-main/bin/python -e '.[mcp,dbos]'

# fastmcp 4 stack (mcp 2.x); httpx is installed only so probe_401_type.py
# can run rote's own 401 predicate, exactly as rote's mcp extra would
uv venv /tmp/rote-probe-fm4
uv pip install --python /tmp/rote-probe-fm4/bin/python 'fastmcp[tasks]==4.1.1' 'httpx>=0.28'

# TS SDKs (exact versions in ts/package.json)
(cd $P/ts && npm install)
```

## Probes

| Appendix | Command | What it shows |
| --- | --- | --- |
| A1 | `python3 -I $P/probe_meta.py https://mcp.linear.app/mcp https://mcp.atlassian.com/v1/mcp ...` | RFC 9728 / 8414 metadata of hosted servers: issuer, DCR endpoint, `client_id_metadata_document_supported`, `authorization_response_iss_parameter_supported` |
| A2 | `/tmp/rote-probe-main/bin/python -I $P/probe_login_path.py linear=https://mcp.linear.app/mcp atlassian=https://mcp.atlassian.com/v1/mcp` | Which registration path `rote mcp login` takes today, and with a CIMD URL |
| A3 | `python3 -I $P/probe_stateless.py <url>` and `python3 -I $P/probe_modern.py <url>` | Raw wire behavior: no-initialize requests, spec-shaped 2026-07-28 requests, `server/discover` |
| A4 | `/tmp/rote-probe-main/bin/python -I $P/probe_rote_client.py` | rote's emitted Python helper (`call_mcp_tool`) against Cloudflare docs, Context7, DeepWiki |
| A4b | `/tmp/rote-probe-fm4/bin/python -I $P/probe_fm4_client.py` | fastmcp 4 `auto` negotiation against the same servers plus GitMCP |
| A5 | `(cd $P/ts && PORT=8799 node modern_only_server.mjs) &`, then `/tmp/rote-probe-main/bin/python -I $P/probe_modern_only.py http://127.0.0.1:8799/mcp`, then the same with `/tmp/rote-probe-fm4/bin/python ... --fm4`, then `(cd $P/ts && node probe_modern_only.mjs http://127.0.0.1:8799/mcp)` | A modern-only server (TS v2, `legacy: "reject"`) against each client |
| A6 | `(cd $P/ts && node probe.mjs)` | TS SDK v1, v2 default, and v2 `auto` against public servers |
| A7 | `/tmp/rote-probe-main/bin/python -I $P/probe_dbos_root.py` and `$P/probe_payload_type.py` | A typed `outputSchema` result through a real DBOS step (SQLite, temp dir) |
| A8 | `probe_401_type.py`, run under both venvs | The exception type a 401 surfaces as, and whether rote's `_find_http_401` matches it |
| A9 | see below | `rote serve` (FastMCP 3.x) as seen by modern clients |

A9 needs a registry file. Generate it from the template, then start
`rote serve` and point the clients at it:

```sh
sed "s#__REPO__#$(pwd)#" $P/serve-registry.template.json > /tmp/rote-probe-registry.json
/tmp/rote-probe-main/bin/rote serve --registry /tmp/rote-probe-registry.json --http --port 8798 &
python3 -I $P/probe_modern.py http://127.0.0.1:8798/mcp
/tmp/rote-probe-fm4/bin/python -I $P/probe_serve_fm4.py http://127.0.0.1:8798/mcp
(cd $P/ts && node probe_serve.mjs http://127.0.0.1:8798/mcp)
```

Every Python probe that touches rote's MCP code points `ROTE_MCP_CONFIG` and
`ROTE_MCP_TOKEN_DIR` at a fresh temp dir first. A stored login on the
machine running the probe is never read or sent.
