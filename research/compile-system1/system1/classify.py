"""System-1 prototype: a small-model step classifier (EXPERIMENTAL).

Two modes, both one schema-locked `claude -p --json-schema` call per skill:

* ``hinted``: the model sees node-kinds.md plus the deterministic
  extractor's units (section, text, MANDATORY, MCP mentions, the
  extractor's own proposal) and returns, per unit, the node list it
  should become, each with a kind and a confidence.
* ``raw``: the model sees node-kinds.md plus SKILL.md (and references)
  and returns the node list with the exact heading each came from.

Uses the same context-isolation flags as rote's emitted judges (imported,
not copied), so the measured cost is the classifier's, not Claude Code's.

usage: python classify.py <hinted|raw> <model> <skill-dir> <out.json>
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from rote.inference._runtime_helper import CLI_CONTEXT_ISOLATION_FLAGS, build_subscription_env

sys.path.insert(0, str(Path(__file__).parent))
from extract import extract  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
RUBRIC = (REPO / "skills" / "rote-compile" / "references" / "node-kinds.md").read_text()
KINDS = ["pure_function", "external_call", "llm_judge", "agent_loop", "hitl_gate"]

_NODE = {
    "type": "object",
    "properties": {
        "id": {"type": "string", "description": "snake_case node id"},
        "kind": {"type": "string", "enum": KINDS},
        "confidence": {"type": "number", "description": "0.0 to 1.0: how sure the kind is right"},
        "reason": {"type": "string", "description": "one short clause"},
    },
    "required": ["id", "kind", "confidence", "reason"],
    "additionalProperties": False,
}

SCHEMAS = {
    "hinted": {
        "type": "object",
        "properties": {
            "units": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "unit": {"type": "integer"},
                        "nodes": {"type": "array", "items": _NODE},
                    },
                    "required": ["unit", "nodes"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["units"],
        "additionalProperties": False,
    },
    "raw": {
        "type": "object",
        "properties": {
            "nodes": {
                "type": "array",
                "items": {
                    **_NODE,
                    "properties": {
                        **_NODE["properties"],
                        "section": {
                            "type": "string",
                            "description": "the exact SKILL.md heading text",
                        },
                    },
                    "required": [*_NODE["required"], "section"],
                },
            }
        },
        "required": ["nodes"],
        "additionalProperties": False,
    },
}

SYSTEM = (
    "You classify the steps of an AI skill into workflow node kinds for the rote "
    "compiler. Follow the rubric exactly. Answer only through the schema."
)

TASK = """\
Below is the rote node-kind rubric, then {what}.

Decide which workflow nodes each step becomes. A step may become several
nodes (for example: a bare tool call, then the pure_function or llm_judge
that processes its result, per the rubric's split rule) or none (pure
narration, or a step that only states the pipeline's input). Prefer the
more deterministic kind when a step could go two ways. Give each node a
confidence: 0.9+ only when the rubric's signals are unambiguous.

=== RUBRIC (node-kinds.md) ===
{rubric}

=== {title} ===
{body}
"""


def build_prompt(mode: str, skill_dir: Path) -> str:
    if mode == "hinted":
        ex = extract(skill_dir)
        units = []
        for i, u in enumerate(ex.units):
            units.append(
                {
                    "unit": i,
                    "section": u.section,
                    "title": u.title,
                    "mandatory_marker": u.mandatory,
                    "mcp_mentions": u.mcp,
                    "deterministic_guess": [p.kind for p in u.proposals],
                    "text": u.text,
                }
            )
        body = json.dumps(
            {"skill": ex.name, "description": ex.description, "units": units}, indent=1
        )
        return TASK.format(
            what=(
                "the skill's steps as a deterministic parser split them, each with "
                "the parser's own guess (which may be wrong)"
            ),
            rubric=RUBRIC,
            title="STEPS",
            body=body,
        )
    parts = [(skill_dir / "SKILL.md").read_text()]
    for ref in (
        sorted((skill_dir / "references").glob("*.md"))
        if (skill_dir / "references").is_dir()
        else []
    ):
        parts.append(f"--- references/{ref.name} ---\n{ref.read_text()}")
    return TASK.format(
        what=(
            "the skill itself (SKILL.md, then its reference files). Use the SKILL.md "
            "heading each node comes from as its section"
        ),
        rubric=RUBRIC,
        title="SKILL",
        body="\n\n".join(parts),
    )


def run(mode: str, model: str, skill_dir: Path) -> dict:
    binary = shutil.which("claude")
    if binary is None:
        raise SystemExit("claude CLI not on PATH")
    prompt = build_prompt(mode, skill_dir)
    command = [
        binary,
        "-p",
        "--output-format",
        "json",
        "--json-schema",
        json.dumps(SCHEMAS[mode]),
        "--model",
        model,
        "--system-prompt",
        SYSTEM,
        *CLI_CONTEXT_ISOLATION_FLAGS,
        "--tools",
        "",
    ]
    t0 = time.monotonic()
    done = subprocess.run(
        command,
        input=prompt,
        capture_output=True,
        text=True,
        env=build_subscription_env(),
        timeout=900,
    )
    wall = time.monotonic() - t0
    env = json.loads(done.stdout)
    if env.get("is_error"):
        raise SystemExit(f"claude -p failed: {str(env.get('result'))[:400]}")
    if env.get("structured_output") is None:
        raise SystemExit(f"no structured output: {str(env.get('result'))[:400]}")
    return {
        "mode": mode,
        "model": model,
        "skill": skill_dir.parent.name,
        "wall_s": round(wall, 1),
        "cost_usd": env.get("total_cost_usd"),
        "usage": env.get("usage"),
        "model_usage": env.get("modelUsage"),
        "prompt_chars": len(prompt),
        "output": env["structured_output"],
    }


if __name__ == "__main__":
    mode, model, skill, out = sys.argv[1], sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4])
    result = run(mode, model, skill)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(
        json.dumps(
            {k: result[k] for k in ("mode", "model", "skill", "wall_s", "cost_usd", "prompt_chars")}
        )
    )
