"""Data sources to auto-pull:: direct-API backend stub.

rote wrote this from the IR (no model). The pipeline's default backend
does not import it; `rote emit --backend api` does.

Returns a value matching this JSON Schema:

{
 "description": "Matching Gmail threads, newest first (inferred from tool description)",
 "items": {
  "properties": {
   "date": {
    "description": "Date of the thread",
    "type": "string"
   },
   "id": {
    "description": "Thread id",
    "type": "string"
   },
   "sender": {
    "description": "Sender of the thread",
    "type": "string"
   },
   "snippet": {
    "description": "Snippet of the latest message",
    "type": "string"
   },
   "subject": {
    "description": "Thread subject",
    "type": "string"
   }
  },
  "required": [
   "id",
   "subject",
   "sender",
   "date",
   "snippet"
  ],
  "type": "object"
 },
 "type": "array"
}
"""

from __future__ import annotations

from typing import Any


def dock_emails_search(request: dict, dock_email_query: str = 'label:dock-activity newer_than:1d') -> list:
    raise NotImplementedError("direct API for gmail.search_threads is not written: no recorded traffic shows its request and response, and rote does not write vendor calls from memory. The default MCP backend makes this call")
