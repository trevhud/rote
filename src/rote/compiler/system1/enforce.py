"""Hold the text agent to System-1's decisions. No inference.

After every stop, the agent's ``pipeline.yaml`` is reconciled with the
skeleton System-1 wrote: decision fields come back from the skeleton,
text fields stay as the agent wrote them. Nodes the agent added are
removed and nodes it dropped are restored. Every change is recorded, and
every change the agent wanted lands in the report as a proposal for a
person, never in the IR.

The agent may also write ``system1/proposals.yaml`` (a list of
``{node, field, value, reason}``). Proposals are checked mechanically
(does the node exist, is the field a decision, would the IR still load
with it) and listed in the report with that verdict. They are not applied:
applying one would be an LLM making a decision.
"""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from typing import Any

import yaml

from rote.compiler.system1.skeleton import BRIEF_DIRNAME, SKELETON_FILENAME
from rote.ir import Pipeline

PROPOSALS_FILENAME = "proposals.yaml"
ENFORCEMENT_FILENAME = "enforcement.json"
INVALID_AGENT_PIPELINE = "agent-pipeline.invalid.yaml"
TEXT_FILENAME = "text.yaml"

#: Node fields that are decisions. Every other field on a node is the
#: agent's to write and is carried over as it wrote it, so a field the IR
#: gains later is never dropped silently (tests/test_system1_hardening.py
#: walks Node.model_fields).
DECISION_FIELDS = (
    "kind",
    "signature",
    "termination",
    "cache",
    "inputs",
    "fan_out",
    "mandatory",
    "constants",
    "impl",
    "mcp",
    "tools",
    "tool_servers",
    "loop_body",
    "signal",
    "notify",
    "timeout",
    "retry",
    "source",
)
#: Pipeline fields that are decisions.
PIPELINE_DECISIONS = ("name", "input", "edges", "entry_nodes", "exit_nodes", "config")


def _load(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _merge_output(skeleton: Any, agent: Any) -> Any:
    """Skeleton output keys are data flow (a fan_out reads ``items``): keep them,
    with the agent's type strings. A plain skeleton output takes the agent's text."""
    if isinstance(skeleton, dict):
        typed = agent if isinstance(agent, dict) else {}
        return {k: typed.get(k, v) for k, v in skeleton.items()}
    return agent if isinstance(agent, (str, dict)) else skeleton


def _types(schema: Any) -> set[str] | None:
    """A schema's declared JSON types, or None when it declares none.

    ``anyOf`` / ``oneOf`` branches contribute their types; one untyped
    branch makes the whole schema untyped (it accepts anything)."""
    if not isinstance(schema, dict):
        return None
    kind = schema.get("type")
    if isinstance(kind, str):
        return {kind}
    if isinstance(kind, list):
        return {k for k in kind if isinstance(k, str)} or None
    branches = schema.get("anyOf") or schema.get("oneOf")
    if isinstance(branches, list) and branches:
        found: set[str] = set()
        for branch in branches:
            types = _types(branch)
            if types is None:
                return None
            found |= types
        return found
    return None


def _compatible(decided: Any, agent: Any) -> bool:
    """Whether the agent's type for one input fits what the input really is.

    An untyped decided input accepts any typing, and an untyped agent input
    accepts anything. Otherwise the types must overlap, and an array's items
    must be compatible too. A real review case: the agent typed a fanned
    judge's whole result (a list at run time) as an object, so the emitted
    judge refused every call and nothing at compile time caught it."""
    want, got = _types(decided), _types(agent)
    if want is None or got is None:
        return True
    if not want & got:
        return False
    if "array" in want & got and isinstance(decided, dict) and isinstance(agent, dict):
        return _compatible(decided.get("items"), agent.get("items"))
    return True


def _merge_spec(
    base: Any, agent: Any, node_id: str, changes: list[str], decided: Any = None
) -> Any:
    """A judge's spec: the skeleton's, with each key the agent wrote laid over it.

    The agent writes the prompt and the output schema. The input schema is a
    decision: its properties are exactly the node's ``inputs`` (the IR
    rejects anything else). The agent may type those inputs more precisely,
    as long as each type fits the decided one (:func:`_compatible`). An input
    schema with any other property names, that requires a name the node
    does not have, or that types an input incompatibly, is replaced by the
    skeleton's and recorded.
    A real cloud compile shipped judges whose input schemas required
    ``ticket`` and ``deadline_hours`` for a node whose inputs were
    ``request`` and ``compute_response_deadline``, and failed IR validation
    after the agent's last stop."""
    if not isinstance(agent, dict):
        return copy.deepcopy(base)
    merged = {**copy.deepcopy(base), **copy.deepcopy(agent)}
    if decided is None:
        decided = base.get("input_schema") if isinstance(base, dict) else None
    mine = agent.get("input_schema")
    if isinstance(decided, dict) and "input_schema" in agent:
        names = set(decided.get("properties") or {})
        props = mine.get("properties") if isinstance(mine, dict) else None
        required = mine.get("required") if isinstance(mine, dict) else None
        decided_props = decided.get("properties") or {}
        fits = (
            isinstance(props, dict)
            and set(props) == names
            and (required is None or (isinstance(required, list) and set(required) <= names))
            and all(_compatible(decided_props[k], props[k]) for k in names)
        )
        if not fits:
            merged["input_schema"] = copy.deepcopy(decided)
            got = sorted(props) if isinstance(props, dict) else props
            changes.append(
                f"`{node_id}` signature_spec.input_schema: agent wrote properties {got!r}, "
                f"kept the node's inputs {sorted(names)!r}"
            )
    return merged


def enforce(work_dir: Path, *, final: bool = False) -> list[str]:
    """Restore System-1's decisions in ``work_dir/pipeline.yaml``; return what changed.

    A ``pipeline.yaml`` the agent left unparseable (a real compile wrote a
    description with an unquoted colon) is left alone during repair rounds,
    so verification reports the YAML error to the agent like any other
    problem. On the ``final`` pass it is kept beside the skeleton for
    review and the skeleton is restored: the decisions never depended on it.
    """
    skeleton = _load(work_dir / BRIEF_DIRNAME / SKELETON_FILENAME)
    path = work_dir / "pipeline.yaml"
    changes: list[str] = []
    try:
        agent = _load(path) if path.is_file() else {}
    except yaml.YAMLError as err:
        if not final:
            return _record(work_dir, ["pipeline.yaml is not valid YAML; sent back for repair"])
        shutil.copyfile(path, work_dir / BRIEF_DIRNAME / INVALID_AGENT_PIPELINE)
        shutil.copyfile(work_dir / BRIEF_DIRNAME / SKELETON_FILENAME, path)
        detail = " ".join(str(err).split())[:160]
        return _record(
            work_dir,
            [
                f"the agent's pipeline.yaml did not parse ({detail}); its text fields were "
                f"lost and the skeleton restored (kept at {BRIEF_DIRNAME}/{INVALID_AGENT_PIPELINE})"
            ],
        )
    agent_nodes = {n.get("id"): n for n in agent.get("nodes", []) if isinstance(n, dict)}
    text = _text_fields(work_dir)
    merged_nodes = []
    for sk in skeleton["nodes"]:
        mine = agent_nodes.get(sk["id"])
        if mine is None:
            changes.append(f"restored node `{sk['id']}` the agent removed")
            merged_nodes.append(copy.deepcopy(sk))
            continue
        # Everything that is not a decision is the agent's: carried as written,
        # falling back to the skeleton's text where the agent left it out.
        decided = _decided_fields(sk)
        node: dict[str, Any] = {
            k: copy.deepcopy(v) for k, v in sk.items() if k not in decided and k != "id"
        }
        node.update(
            {k: v for k, v in mine.items() if k not in decided and k not in ("id", "output")}
        )
        node["id"] = sk["id"]
        for key in decided:
            if key in sk:
                node[key] = copy.deepcopy(sk[key])
            if mine.get(key) != sk.get(key):
                changes.append(
                    f"`{sk['id']}` {key}: agent wrote {mine.get(key)!r}, kept {sk.get(key)!r}"
                )
        node["output"] = _merge_output(sk.get("output"), mine.get("output"))
        if "signature_spec" in sk:
            node["signature_spec"] = _merge_spec(
                sk["signature_spec"], mine.get("signature_spec"), sk["id"], changes
            )
        for key, value in text.get(sk["id"], {}).items():
            if key == "output":
                node["output"] = _merge_output(sk.get("output"), value)
            elif key == "signature_spec" and "signature_spec" in sk:
                # Checked against the skeleton's decided schema, not against
                # whatever pipeline.yaml's copy already says.
                node["signature_spec"] = _merge_spec(
                    node["signature_spec"],
                    value,
                    sk["id"],
                    changes,
                    sk["signature_spec"].get("input_schema"),
                )
            elif key not in decided and key != "id":
                node[key] = value
        merged_nodes.append(node)
    for extra in set(agent_nodes) - {n["id"] for n in skeleton["nodes"]}:
        changes.append(f"removed node `{extra}` the agent added")
    merged = {k: v for k, v in agent.items() if k not in ("nodes", *PIPELINE_DECISIONS)}
    for key in PIPELINE_DECISIONS:
        if key in skeleton:
            merged[key] = copy.deepcopy(skeleton[key])
            if agent.get(key) != skeleton[key]:
                changes.append(f"pipeline {key}: restored the skeleton's")
    merged.setdefault("version", skeleton.get("version", "0.1.0"))
    merged["nodes"] = merged_nodes
    path.write_text(yaml.safe_dump(merged, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return _record(work_dir, changes)


#: Decisions that rules make only when the skill states them. When the
#: skeleton leaves one unset, the skill gave nothing to decide from, and the
#: text agent writes it from the skill's own wording (a loop's stop condition
#: and bound). Restoring the skeleton's "unset" would ship a loop with no stop.
#: A gate's ``notify`` is the structure pass's message when it wrote one; the
#: rules path writes none, and the text agent's stays (the IR checks it).
_WHEN_STATED = ("termination", "notify")


def _decided_fields(skeleton_node: dict[str, Any]) -> tuple[str, ...]:
    return tuple(k for k in DECISION_FIELDS if k not in _WHEN_STATED or k in skeleton_node)


def _text_fields(work_dir: Path) -> dict[str, dict[str, Any]]:
    """``system1/text.yaml``: the agent's text per node, written once instead of
    editing ``pipeline.yaml`` field by field (a real run spent 9 turns on Edits)."""
    path = work_dir / BRIEF_DIRNAME / TEXT_FILENAME
    if not path.is_file():
        return {}
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError:
        return {}
    if not isinstance(loaded, dict):
        return {}
    return {str(k): v for k, v in loaded.items() if isinstance(v, dict)}


def _record(work_dir: Path, changes: list[str]) -> list[str]:
    record = work_dir / BRIEF_DIRNAME / ENFORCEMENT_FILENAME
    previous = json.loads(record.read_text()) if record.is_file() else []
    record.write_text(json.dumps([*previous, changes], indent=1), encoding="utf-8")
    return changes


def all_changes(work_dir: Path) -> list[str]:
    """Every decision the agent edited across all stops, first occurrence order."""
    record = work_dir / BRIEF_DIRNAME / ENFORCEMENT_FILENAME
    rounds = json.loads(record.read_text()) if record.is_file() else []
    seen: list[str] = []
    for changes in rounds:
        seen += [c for c in changes if c not in seen]
    return seen


def check_proposals(work_dir: Path) -> list[str]:
    """Each proposal with a mechanical verdict. Nothing is applied."""
    path = work_dir / BRIEF_DIRNAME / PROPOSALS_FILENAME
    if not path.is_file():
        return []
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as err:
        # A malformed file must cost nothing: it is reported, not raised.
        detail = " ".join(str(err).split())[:160]
        return [f"{BRIEF_DIRNAME}/{PROPOSALS_FILENAME} did not parse, so none was read ({detail})"]
    if loaded is not None and not isinstance(loaded, list):
        return [f"{BRIEF_DIRNAME}/{PROPOSALS_FILENAME} is not a list, so none was read"]
    proposals = loaded or []
    skeleton = _load(work_dir / BRIEF_DIRNAME / SKELETON_FILENAME)
    ids = {n["id"] for n in skeleton["nodes"]}
    out = []
    for p in proposals:
        if not isinstance(p, dict):
            continue
        node, fld, value = p.get("node"), p.get("field"), p.get("value")
        reason = str(p.get("reason", "")).strip()
        if node not in ids:
            verdict = "rejected: no such node"
        elif fld not in DECISION_FIELDS:
            verdict = "rejected: not a decision field"
        else:
            trial = copy.deepcopy(skeleton)
            for n in trial["nodes"]:
                if n["id"] == node:
                    n[fld] = value
            try:
                Pipeline.model_validate(trial)
                verdict = "valid IR if applied"
            except ValueError as err:
                verdict = f"invalid IR if applied: {str(err).splitlines()[1:2]}"
        out.append(f"`{node}` {fld} -> {value!r} ({verdict}). Agent's reason: {reason}")
    return out
