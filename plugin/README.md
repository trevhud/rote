# rote for Claude

rote turns an Anthropic-style skill (a SKILL.md plus optional references/)
into a deterministic workflow. A compiler agent reads the skill, classifies
each step, extracts the steps that never needed a model into plain code, and
keeps typed LLM calls only where judgment is required. The result runs as a
durable background workflow on DBOS, Temporal, Cloudflare Workflows, Inngest,
DBOS TypeScript, or a plain Python script, instead of a 10 to 20 minute agent
loop.

This plugin adds skills to Claude. Each one lives in its own folder under
[`skills/`](skills/). The main ones:

- **compile** (`/rote:compile`): confirms the skill directory with you, asks
  which runtime you want, runs the rote CLI in the background, and explains
  the emitted pipeline.
- **serve** (`/rote:serve`): registers a compiled, deployed pipeline and wires
  `rote serve` into Claude as an MCP server, so Claude can start a run, poll
  it, and deliver approvals at human-in-the-loop gates.

## Install

In a Claude Code session (v2.1.275 or later):

```
/plugin install rote --marketplace trevhud/rote
```

From your shell (v2.1.292 or later):

```
claude plugin install rote --marketplace trevhud/rote
```

On older versions, add the marketplace first:

```
/plugin marketplace add trevhud/rote
/plugin install rote@rote
```

These forms and version requirements come from
[Install and manage plugins](https://code.claude.com/docs/en/discover-plugins#add-a-marketplace-and-install-in-one-command).

## Try it

- "Compile the skill in ./skills/bdr-outreach into a DBOS workflow."
- "This weekly-report skill is slow and expensive. Make it deterministic."
- "Register my compiled pipeline in ./bdr-out and let me trigger it from Claude."

## Requirements

uv (the skills run the CLI with `uvx --from rote-cli rote`), and one compiler
driver: the `claude` CLI, the `codex` CLI, or an Anthropic or OpenAI API key.

## What it runs, sends, and fetches

- Downloads the `rote-cli` package from PyPI through uv.
- Sends the skill's text to the compiler model you choose: Claude through your
  own `claude` session, Codex through your `codex` session, or the Anthropic or
  OpenAI API with your own key.
- With the API drivers, calls read-only tools on MCP servers you have
  registered with `rote mcp add` and authenticated, so the compiler can observe
  real tool output. Pass `--no-mcp` to turn this off.
- Only if you have run `rote login`: compilation runs on rote cloud
  (app.roteskills.com), which receives the skill bundle and hosts the deployed
  workflow. Pass `--local` to keep everything on your machine.
- Writes output only to the `--out` directory you choose. Stores credentials
  locally under `~/.local/share/rote/` and `~/.rote/`. No telemetry.

## Troubleshooting

- `uvx: command not found`: install uv from https://docs.astral.sh/uv/.
- A DBOS run stays `enqueued`: the emitted app is not running against the
  registered system database. Start it with `python main.py --serve`.
- A new pipeline does not appear in Claude Desktop or claude.ai: those
  clients read the tool list at connect time. Reconnect the server.

## Support

Issues: https://github.com/trevhud/rote/issues
Security: see [SECURITY.md](https://github.com/trevhud/rote/blob/main/SECURITY.md).
License: Apache-2.0.
