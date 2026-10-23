"""Data sources to auto-pull:: direct-API backend stub.

rote wrote this from the IR (no model). The pipeline's default backend
does not import it; `rote emit --backend api` does.

Returns a value matching this JSON Schema:

{
 "description": "CSV text of the Dwell Log tab, header row first, including the Open/Closed column",
 "type": "string"
}
"""

from __future__ import annotations

from typing import Any


def open_dwell_read(request: dict, dwell_ticket_file_id: str = '1ExampleDwellTicketLogFileId0000000000000000', dwell_ticket_tab: str = 'Dwell Log - Form Responses') -> str:
    raise NotImplementedError("direct API for google_drive.read_file_content is not written: no recorded traffic shows its request and response, and rote does not write vendor calls from memory. The default MCP backend makes this call")
