"""1. Search Microsoft Learn: direct-API backend stub.

rote wrote this from the IR (no model). The pipeline's default backend
does not import it; `rote emit --backend api` does.

Returns a value matching this JSON Schema:

{
 "properties": {
  "results": {
   "items": {
    "properties": {
     "content": {
      "type": [
       "string",
       "null"
      ]
     },
     "contentUrl": {
      "type": "string"
     },
     "content_omitted": {
      "type": "boolean"
     },
     "extensionData": {
      "type": [
       "object",
       "null"
      ]
     },
     "id": {
      "type": [
       "string",
       "null"
      ]
     },
     "title": {
      "type": "string"
     }
    },
    "type": "object"
   },
   "type": "array"
  }
 },
 "type": "object"
}
"""

from __future__ import annotations

from typing import Any


def search_microsoft_learn(request: dict, arg_query: str) -> dict:
    raise NotImplementedError("direct API for mslearn.microsoft_docs_search is not written: no recorded traffic shows its request and response, and rote does not write vendor calls from memory. The default MCP backend makes this call")
