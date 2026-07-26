"""``rote serve`` — expose compiled pipelines as MCP tools.

Three modules:

* :mod:`rote.serve.registry` — the manifest registry (``~/.rote/registry.json``)
  mapping tool names to compiled pipelines and their runtime trigger config.
* :mod:`rote.serve.backends` — runtime trigger backends (Temporal, Cloudflare)
  that start a compiled workflow and poll its status.
* :mod:`rote.serve.server` — the FastMCP server that sources one MCP tool
  (plus a ``<tool>_status`` companion) per registry entry.

The heavy dependency (``fastmcp``) is only imported by :mod:`rote.serve.server`
so that ``rote register`` works without the ``serve`` extra installed.
"""
