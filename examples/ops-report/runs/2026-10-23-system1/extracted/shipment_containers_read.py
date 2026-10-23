"""Data sources to auto-pull:: direct-API backend stub.

rote wrote this from the IR (no model). The pipeline's default backend
does not import it; `rote emit --backend api` does.

Returns a value matching this JSON Schema:

{
 "description": "CSV text of the sheet, inferred from the tool returning file content as text",
 "type": "string"
}
"""

from __future__ import annotations

from typing import Any


def shipment_containers_read(request: dict, shipment_containers_file_id: str = '1ExampleShipmentContainersFileId00000000000') -> str:
    raise NotImplementedError("direct API for google_drive.read_file_content is not written: no recorded traffic shows its request and response, and rote does not write vendor calls from memory. The default MCP backend makes this call")
