"""3. Read the repo's wiki outline: direct-API backend stub.

rote wrote this from the IR (no model). The pipeline's default backend
does not import it; `rote emit --backend api` does.

Returns a value matching this JSON Schema:

{
 "type": "string"
}
"""

from __future__ import annotations

from typing import Any


def read_repo_wiki_outline(request: dict, arg_repoName: str) -> str:
    raise NotImplementedError("direct API for deepwiki.read_wiki_structure is not written: no recorded traffic shows its request and response, and rote does not write vendor calls from memory. The default MCP backend makes this call")
