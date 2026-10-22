"""The person at a skill's gates, as an MCP tool the agent calls when it asks.

``python -m rote.eval.gate_server <answers.json> <calls.jsonl>``

A skill run as an agent (``rote run <skill> --signal``, and the compile
bench's fidelity references) used to get every gate answer in its prompt up
front. It then knew each approval before it reached the step that asks for
it, and its trace had no moment at which it asked. This stdio MCP server
holds the answers instead: the agent calls ``ask_person`` when the skill
would ask someone, and the i-th call gets the i-th answer. Each call is
appended to ``calls.jsonl`` in the call log's line format, so in a fidelity
recording the reads before each ask are the reads the skill makes before
that gate.

Stdlib only, newline-delimited JSON-RPC on stdin and stdout, launched by the
agent's own MCP client: no port, and no MCP SDK needed to run a skill.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

#: The server and tool names a run's MCP config and allowlist use.
GATE_SERVER = "rote_gates"
GATE_TOOL = "ask_person"

_TOOL = {
    "name": GATE_TOOL,
    "description": (
        "Ask the person this skill needs an answer or an approval from. Call it "
        "at the moment the skill asks them, with what you would ask; the "
        "result is their answer."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "What you would ask the person"}
        },
        "required": ["question"],
    },
    "annotations": {"readOnlyHint": True},
}


class GateAnswers:
    """The answers, served in order, and a log line for every ask."""

    def __init__(self, answers: list[Any], log_path: Path) -> None:
        self.answers = answers
        self.log_path = log_path
        self.asked = 0

    def ask(self, arguments: dict[str, Any]) -> tuple[str, bool]:
        """``(text, is_error)`` for one ``ask_person`` call."""
        index = self.asked
        self.asked += 1
        answered = index < len(self.answers)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "ts": time.time(),
                        "server": GATE_SERVER,
                        "tool": GATE_TOOL,
                        "arguments": arguments,
                        "side_effect": False,
                        "matched": answered,
                        "response": index if answered else None,
                        "gate": index,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
        if not answered:
            return f"no one else to ask: all {len(self.answers)} answer(s) were given", True
        return json.dumps(self.answers[index]), False


def _reply(request_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def handle(message: dict[str, Any], gates: GateAnswers) -> dict[str, Any] | None:
    """The response to one JSON-RPC message, or None for a notification."""
    method = message.get("method")
    request_id = message.get("id")
    if request_id is None:
        return None  # notifications/initialized and the like
    if method == "initialize":
        params = message.get("params") or {}
        return _reply(
            request_id,
            {
                "protocolVersion": params.get("protocolVersion", "2025-06-18"),
                "capabilities": {"tools": {}},
                "serverInfo": {"name": GATE_SERVER, "version": "1"},
            },
        )
    if method == "ping":
        return _reply(request_id, {})
    if method == "tools/list":
        return _reply(request_id, {"tools": [_TOOL]})
    if method == "tools/call":
        params = message.get("params") or {}
        if params.get("name") != GATE_TOOL:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32602, "message": f"unknown tool {params.get('name')!r}"},
            }
        text, is_error = gates.ask(params.get("arguments") or {})
        return _reply(
            request_id, {"content": [{"type": "text", "text": text}], "isError": is_error}
        )
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": -32601, "message": f"method not found: {method}"},
    }


def main(argv: list[str]) -> None:
    answers = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
    gates = GateAnswers(list(answers), Path(argv[1]))
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        response = handle(json.loads(line), gates)
        if response is not None:
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main(sys.argv[1:])
