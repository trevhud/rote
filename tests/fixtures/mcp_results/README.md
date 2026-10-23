`docs_digest.json` holds two real MCP tools as of 2026-10-20: the
`declared` entries are the live `tools/list` input and output schemas of
DeepWiki's `read_wiki_structure` (https://mcp.deepwiki.com/mcp) and
Microsoft Learn's `microsoft_docs_search` (https://learn.microsoft.com/api/mcp),
and the `observations` are what a `rote compile --baseline` run of
the docs-digest skill (now `examples/docs-digest/skill/`) recorded for one
call each, page text shortened.

DeepWiki is a fastmcp server: its tool returns a string, which arrives
wrapped as `{"result": ...}` under an `x-fastmcp-wrap-result` output
schema. Microsoft Learn returns an object.
