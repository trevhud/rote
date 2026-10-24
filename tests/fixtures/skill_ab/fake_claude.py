#!/usr/bin/env python3
"""A stand-in for ``claude -p`` that behaves like an agent, with no model.

``rote eval --run --runtime skill``'s tests run the whole harness with this
as the executable. It reads the argv the harness built and acts on it:

- the control prompt: one request, no tool;
- the original side (no ``Bash(rote skill run ...)`` rule): asks the person
  through the ``rote_gates`` server when the run has one, then writes
  ``$FAKE_ORIGINAL_OUTPUT`` (a JSON file) as its result.json;
- the compiled side: runs ``rote skill run <name> start '<input>'`` the way
  the compiled SKILL.md says, relays each HUMAN block to the person (the
  gate server) and answers with ``--said``, then copies the driver's
  result into its own result.json, plus ``$FAKE_EXTRA`` (JSON) on top.

``$FAKE_MODE`` makes it misbehave: ``bypass`` writes a result without
running the driver; ``self_answer`` answers a gate without asking anyone.

It prints a stream-json transcript whose usage is fixed and known, so the
tests can assert the parsed numbers exactly: request i reports input
``10*i``, cache write ``100*i``, cache read ``1000*i`` on each of its two
content blocks (Claude Code repeats usage per block).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

argv = sys.argv[1:]


def flag(name: str) -> str:
    return argv[argv.index(name) + 1]


prompt = flag("-p")
allowed = flag("--allowedTools")
mcp = json.loads(Path(flag("--mcp-config")).read_text())["mcpServers"]
requests = 0
said_log: list[str] = []


def ask_person(question: str) -> str:
    gate = mcp["rote_gates"]
    message = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": "ask_person", "arguments": {"question": question}},
    }
    proc = subprocess.run(
        [gate["command"], *gate["args"]],
        input=json.dumps(message) + "\n",
        capture_output=True,
        text=True,
        check=True,
    )
    return str(json.loads(proc.stdout)["result"]["content"][0]["text"])


def rote(*args: str) -> str:
    global requests
    requests += 1
    proc = subprocess.run(["rote", *args], capture_output=True, text=True)
    return proc.stdout


def header(block: str) -> dict[str, str]:
    first = block.splitlines()[0].split()
    fields = {"kind": first[1]}
    fields.update(dict(part.split("=", 1) for part in first[3:] if "=" in part))
    return fields


def main() -> None:
    global requests
    mode = os.environ.get("FAKE_MODE", "honest")
    if prompt.startswith("Reply with the single word ok"):
        requests = 1
    elif "Bash(rote skill run " not in allowed:
        requests = 1
        if "rote_gates" in mcp:
            ask_person("Approve?")
            requests += 1
        output = json.loads(Path(os.environ["FAKE_ORIGINAL_OUTPUT"]).read_text())
        Path("result.json").write_text(json.dumps(output))
        requests += 1
    else:
        name = re.search(r"Bash\(rote skill run ([a-z0-9-]+) \*\)", allowed).group(1)  # type: ignore[union-attr]
        payload = json.loads(re.search(r"```json\n(.*?)\n```", prompt, re.S).group(1))  # type: ignore[union-attr]
        extra = json.loads(os.environ.get("FAKE_EXTRA") or "{}")
        if mode == "bypass":
            Path("result.json").write_text(json.dumps(extra))
            requests = 1
        else:
            block = rote("skill", "run", name, "start", json.dumps(payload))
            while block.startswith("ROTE/1 HUMAN"):
                fields = header(block)
                if mode == "self_answer":
                    words = json.dumps(json.loads(os.environ["FAKE_SELF_ANSWER"]))
                else:
                    words = ask_person(block)
                    requests += 1
                lines = block.splitlines()
                next_line = [line for line in lines if line.startswith("rote skill run")]
                assert next_line, block
                block = rote(
                    "skill",
                    "run",
                    name,
                    "answer",
                    fields["run"],
                    fields["step"],
                    words,
                    "--said",
                    words,
                )
            if not block.startswith("ROTE/1 DONE"):
                sys.stderr.write(block)
                raise SystemExit(1)
            path = re.search(r"^(?:result|saved): (/\S+) \(", block, re.M).group(1)  # type: ignore[union-attr]
            result = json.loads(Path(path).read_text())
            merged: dict[str, object] = {}
            for node_result in result.values():
                if isinstance(node_result, dict):
                    merged.update(node_result)
            merged.update(extra)
            Path("result.json").write_text(json.dumps(merged))
            requests += 1
    for i in range(1, requests + 1):
        usage = {
            "input_tokens": 10 * i,
            "cache_creation_input_tokens": 100 * i,
            "cache_read_input_tokens": 1000 * i,
            "output_tokens": 1,
        }
        for block_type in ("thinking", "text"):
            print(
                json.dumps(
                    {
                        "type": "assistant",
                        "message": {
                            "id": f"msg_{i}",
                            "usage": usage,
                            "content": [{"type": block_type}],
                        },
                    }
                )
            )
    total = sum(range(1, requests + 1))
    print(
        json.dumps(
            {
                "type": "result",
                "is_error": False,
                "num_turns": requests,
                "total_cost_usd": 0.01 * requests,
                "modelUsage": {
                    "fake-model": {
                        "inputTokens": 10 * total,
                        "outputTokens": 7 * requests,
                        "cacheCreationInputTokens": 100 * total,
                        "cacheReadInputTokens": 1000 * total,
                    }
                },
                "permission_denials": [],
            }
        )
    )


main()
