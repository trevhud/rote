# rote's OAuth Client ID Metadata Document

[`rote-cli.json`](rote-cli.json) is the Client ID Metadata Document
(CIMD) that `rote mcp login` offers to authorization servers that
advertise `client_id_metadata_document_supported`. Its URL is the client
id, so it must be served at exactly:

```
https://roteskills.com/oauth/rote-cli.json
```

with `Content-Type: application/json`. That URL is
`rote.mcp.auth.DEFAULT_CLIENT_METADATA_URL`, and the document's
`client_id` must equal it byte for byte (CIMD draft section 3: simple
string comparison).

## What is in it, and why

- `client_name` / `client_uri`: what the consent screen shows. This is
  the trust CIMD adds over DCR: a named client instead of an anonymous
  one minted per login. It is not client authentication. Anyone can
  point an authorization server at this URL, and the spec notes CIMD
  cannot prevent `localhost` redirect impersonation.
- `token_endpoint_auth_method: "none"`: a CLI is a public client. The
  CIMD draft forbids the shared-secret methods.
- `grant_types`: `authorization_code` and `refresh_token`, because the
  emitted workflows refresh unattended.
- `application_type: "native"`, which the spec asks of CLI tools.
- `redirect_uris`: `localhost` and `127.0.0.1` on ports 33418 to 33421.
  A CIMD authorization server matches the redirect URI against this list
  by exact string comparison, and nothing in the CIMD draft or the MCP
  spec lets a loopback port vary. So rote publishes a fixed set and
  `rote mcp login` binds the first free one. The list must stay equal
  to `rote.mcp.auth.callback_redirect_uris()`, which a unit test enforces.
  `127.0.0.1` is listed although the CLI currently redirects to
  `localhost`, so switching hosts later needs no republish.

## Changing it

Authorization servers may cache the document (respecting HTTP cache
headers), so a change can take time to reach them. Add redirect URIs
before the CLI starts using them, never after. Removing a port strands
every installed CLI that still binds it.

## Publishing

The landing site (`rote-cloud/marketing`) serves this file. It is not
fetched by rote itself. Only authorization servers read it. Until it is
published, a login against a CIMD server (Linear, Notion, Sentry) fails,
because that server cannot fetch the client it is asked to authorize. The
document must be live before a release ships this CLI. Until then, a user
can set `ROTE_MCP_CLIENT_METADATA_URL=` (empty) to turn CIMD off and log
in with DCR as before.
