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

The landing site (`rote-cloud/marketing`) serves this file. rote never
offers a document it has not seen: when an authorization server
advertises CIMD, `rote mcp login` fetches the URL once (5 second
timeout, cached per process) and offers it only if the response is 200
and equals the packaged copy (`src/rote/mcp/client_metadata.json`, the
same bytes as this file) with `client_id` set to the URL. Anything else
(404, a stale or edited copy, no network) prints one line and the login
registers a client with DCR instead. So an unpublished or outdated
document degrades to the pre-CIMD behavior rather than breaking logins,
and the release checklist (`docs/releasing.md`) diffs the live copy
against the repo before each release.
