# Example: Docs Digest

The demo that `rote try` and `rote setup` run. A small skill that builds
a reading list on a Microsoft technology from Microsoft Learn and adds
the DeepWiki outline of the matching open-source repo. Both MCP servers
are public and need no login, so the whole thing runs with two
`rote mcp add` lines and nothing else:

| Server | URL | Tool |
| --- | --- | --- |
| `mslearn` | https://learn.microsoft.com/api/mcp | `microsoft_docs_search` |
| `deepwiki` | https://mcp.deepwiki.com/mcp | `read_wiki_structure` |

Run as an agent, the skill costs a few cents of model time per run.
Compiled, it is two MCP calls and two small Python functions, with no
model call at all.

## What it demonstrates

- **MCP-bound steps.** `search_microsoft_learn` and
  `read_repo_wiki_outline` are bare tool calls through the emitted MCP
  helper. They run in the same entry wave, concurrently.
- **Rules lifted out of prose.** "Keep the first five results" and the
  digest's two section names are code in `compiled/extracted/`.
- **Live data.** The servers answer from live indexes, so a run today
  can differ from the recorded one. `reference.json` says which fields
  are expected to drift and why.

## Run it

From the repo root:

```sh
rote mcp add mslearn https://learn.microsoft.com/api/mcp --auth none --read-only
rote mcp add deepwiki https://mcp.deepwiki.com/mcp --auth none --read-only

# The precompiled pipeline (needs the dbos extra: SQLite, no Node, no Docker)
rote emit examples/docs-digest/compiled/pipeline.yaml --runtime dbos --out /tmp/docs-digest-dbos
rote run /tmp/docs-digest-dbos --input examples/docs-digest/skill/input.json

# The skill itself, as an agent, read-only (bills your Claude subscription)
rote run examples/docs-digest/skill --input examples/docs-digest/skill/input.json
```

The python runtime cannot run this pipeline. It has no MCP backend, so
it would call the two tool steps' direct-API modules, and those are stubs
(rote does not write vendor calls from memory). `rote run` refuses it up
front. dbos is the runtime that speaks MCP and needs no Node and no
Docker.

## The recorded reference

`reference.json` (format `rote-demo-reference/1`, documented in
`rote.demo`) holds one real run of each side on `skill/input.json`,
recorded on 2026-10-22:

| | Skill, as an agent | Compile | Pipeline |
| --- | --- | --- | --- |
| Wall time | 18.5 s | 34.4 s | 9.1 s, then 5.4 s |
| Turns or calls | 4 turns | 6 generations + 1 structure call | 0 model calls |
| Cost | $0.088 | $0.079 | $0 |
| Model | claude-sonnet-4-6 | claude-sonnet-5-5, claude-haiku-5-5 | none |

The skill's digest and the pipeline's digest were the same text, except
that the pipeline ends it with a newline. Two pipeline runs five minutes
apart got identical tool results and returned the same digest. The
`drift` section of the reference lists what is still expected to change
over days: the reading list first, the repo outline slowly.

`reference.json` also keeps the two tool results the first pipeline run
received (`pipeline_run.tool_results`, Microsoft Learn's page excerpts
dropped, since the pipeline reads only titles and links), so the
compiled code can be checked offline:
`tests/test_docs_digest_demo.py` feeds them through the extracted
functions and requires the recorded digest back.

## Layout

```
examples/docs-digest/
├── README.md
├── reference.json           # the recorded runs, see above
├── skill/
│   ├── SKILL.md
│   ├── input.json           # the sample input; compile picks it up
│   └── mcp-tools.json       # tools/list snapshot of the two tools
└── compiled/                # `rote compile --runtime dbos` output
    ├── pipeline.yaml
    ├── extracted/
    ├── tests/
    └── compile-report.md
```
