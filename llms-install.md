# Installing rote as an MCP server

This file is for AI agents (Cline, Claude Code, and others) setting up the
`rote serve` MCP server for a user. The human-facing guide is
[docs/mcp-trigger.md](docs/mcp-trigger.md).

## What the server does

`rote serve` exposes every compiled, registered rote pipeline as one MCP tool,
plus two built-in tools: `rote_list_pipelines` and `rote_describe_pipeline`.
With nothing registered yet, only the two built-in tools appear. Pipelines
come from `rote compile <skill>` followed by `rote register <out-dir>`.

## Requirements

- Python 3.12 or newer.
- `uv` (recommended, gives a zero-install launch) or `pip`.
- No API keys. The server triggers workflows the user has already deployed;
  it holds no credentials of its own.

## Launch command (stdio)

```sh
uvx --from 'rote-cli[serve]' rote serve
```

The `serve` extra is required: the base package carries only the MCP client
half. With pip instead of uv: `pip install 'rote-cli[serve]'` then `rote serve`.

## MCP client configuration

Generic stdio config:

```json
{
  "mcpServers": {
    "rote": {
      "command": "uvx",
      "args": ["--from", "rote-cli[serve]", "rote", "serve"]
    }
  }
}
```

Claude Code:

```sh
claude mcp add rote -- uvx --from 'rote-cli[serve]' rote serve
```

## Verifying the install

Call `rote_list_pipelines`. An empty list is a successful install with no
pipelines registered yet. To get a first pipeline, follow the quickstart in
[README.md](README.md): `rote compile`, deploy the emitted runtime, then
`rote register`.
