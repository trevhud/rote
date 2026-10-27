"""Pipeline input fields the text agent adds, checked by code. No inference.

System-1 decides the pipeline input schema and restores it after every
agent stop (:mod:`rote.compiler.system1.enforce`), so a module that needs a
value the caller supplies (a dashboard URL the skill asks for and never
gives) had no way to declare it: a real compile's proposal to add one was
rejected as "not a decision field", and the field stayed undeclared until
the step that read it raised.

This is the one narrow way in. An entry names one field (``name``, its JSON
``type``, ``required``, a one-line ``description``) and the skill text that
motivates it (``quote``, checked like a ``# skill:`` citation). It may only
ADD a property to the input schema, or make a declared, untyped-or-same-typed
property required. It never removes, renames or retypes a field and never
touches nodes, edges or any other decision. Each addition is laid over the
skeleton's schema at every settle, so it survives the restore, and the
result must still load as a :class:`~rote.ir.Pipeline`. Every entry, applied
or refused, is recorded for the compile report.

Two writers, one file: the single-session agent writes
``system1/input_fields.yaml`` itself, and the parallel path writes it from
the module pieces' ``input_fields`` answers.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from rote.compiler.system1.skeleton import BRIEF_DIRNAME
from rote.ir import Pipeline, is_field_name

#: What the agent writes: a YAML list of entries.
FILENAME = "input_fields.yaml"
#: What rote did with each entry at the last settle.
RECORD_FILENAME = "input_fields.json"

JSON_TYPES = ("string", "integer", "number", "boolean", "array", "object")
KEYS = ("name", "type", "required", "description", "quote")
#: A description is one line a reviewer reads, not a document.
MAX_DESCRIPTION = 200

#: One entry, as a strict structured-output schema (generate.ANSWER_SCHEMA).
ENTRY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "type": {"enum": list(JSON_TYPES)},
        "required": {"type": "boolean"},
        "description": {"type": "string"},
        "quote": {"type": "string"},
    },
    "required": list(KEYS),
    "additionalProperties": False,
}


@dataclass
class Outcome:
    """Entries applied, and entries refused with why."""

    applied: list[dict[str, Any]] = field(default_factory=list)
    refused: list[tuple[dict[str, Any], str]] = field(default_factory=list)


@dataclass(frozen=True)
class SkillText:
    """The skill's citable prose and table rows, read once."""

    paragraphs: list[str]
    rows: list[Any]

    @classmethod
    def read(cls, skill_dir: Path) -> SkillText:
        from rote.compiler.provenance import skill_paragraphs, skill_table_rows

        return cls(skill_paragraphs(skill_dir), skill_table_rows(skill_dir))


def _shape_problem(entry: Any) -> str | None:
    if not isinstance(entry, dict):
        return "an entry must be a mapping of " + ", ".join(KEYS)
    if set(entry) != set(KEYS):
        return f"an entry has exactly the keys {', '.join(KEYS)}; got {', '.join(map(str, entry))}"
    name = entry["name"]
    if not isinstance(name, str) or not is_field_name(name):
        return f"name {name!r} must be an identifier (letters, digits, underscores)"
    if entry["type"] not in JSON_TYPES:
        return f"type {entry['type']!r} must be one of {', '.join(JSON_TYPES)}"
    if not isinstance(entry["required"], bool):
        return "required must be true or false"
    description = entry["description"]
    if not isinstance(description, str) or not description.strip():
        return "description must be one line of text"
    if len(description.splitlines()) > 1 or len(description) > MAX_DESCRIPTION:
        return f"description must be one line of at most {MAX_DESCRIPTION} characters"
    if not isinstance(entry["quote"], str):
        return "quote must be the skill's own text"
    return None


def _declared_type(prop: Any) -> str | None:
    kind = prop.get("type") if isinstance(prop, dict) else None
    return kind if isinstance(kind, str) else None


def apply(
    doc: dict[str, Any], entries: list[Any], skill: SkillText, sample_payload: Any = None
) -> Outcome:
    """Lay ``entries`` over ``doc``'s input schema, in order; ``doc`` is changed in place.

    ``sample_payload`` is the compile's sample input: a field it does not
    carry cannot be required, because the sample is a real input and would
    then fail the entry check. An entry repeated exactly (two modules read
    one field) is applied once."""
    from rote.compiler.provenance import citation_problem

    out = Outcome()
    for entry in entries:
        problem = _shape_problem(entry)
        if problem is not None:
            out.refused.append((entry if isinstance(entry, dict) else {"entry": entry}, problem))
            continue
        name, kind, required = entry["name"], entry["type"], entry["required"]
        if any(
            (a["name"], a["type"], a["required"]) == (name, kind, required) for a in out.applied
        ):
            continue
        cited = citation_problem(entry["quote"], skill.paragraphs, skill.rows)
        if cited is not None:
            out.refused.append((entry, f"its quote is not the skill's text: {cited}"))
            continue
        trial = copy.deepcopy(doc)
        container = trial.get("input")
        if not isinstance(container, dict):
            out.refused.append((entry, "the pipeline declares no input to add a field to"))
            continue
        schema = container.setdefault("input_schema", {"type": "object"})
        props = schema.setdefault("properties", {})
        listed = schema.get("required") or []
        if name in props:
            have = _declared_type(props[name])
            if any(a["name"] == name for a in out.applied):
                why = f"`{name}` was already added by an earlier entry with another shape"
            elif not required:
                why = f"`{name}` is already declared; an entry only adds a field"
            elif name in listed:
                why = f"`{name}` is already declared and required"
            elif have is not None and have != kind:
                why = f"`{name}` is declared as {have}, so it cannot be required as {kind}"
            else:
                why = None
            if why is not None:
                out.refused.append((entry, why))
                continue
        else:
            props[name] = {"type": kind, "description": " ".join(entry["description"].split())}
        if required:
            if isinstance(sample_payload, dict) and name not in sample_payload:
                out.refused.append(
                    (
                        entry,
                        f"the sample input has no `{name}`, so it cannot be required; "
                        "add it with required: false",
                    )
                )
                continue
            schema["required"] = [*listed, name]
        try:
            Pipeline.model_validate(trial)
        except ValueError as err:
            first = next((ln for ln in str(err).splitlines()[1:] if ln.strip()), str(err))
            out.refused.append((entry, f"the pipeline would not load with it: {first.strip()}"))
            continue
        doc.clear()
        doc.update(trial)
        out.applied.append(dict(entry))
    return out


def read(work_dir: Path) -> tuple[list[Any], str | None]:
    """The entries in ``system1/input_fields.yaml``, and why none could be read."""
    path = work_dir / BRIEF_DIRNAME / FILENAME
    if not path.is_file():
        return [], None
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as err:
        detail = " ".join(str(err).split())[:160]
        return [], f"{BRIEF_DIRNAME}/{FILENAME} did not parse, so no field was added ({detail})"
    if loaded is None:
        return [], None
    if not isinstance(loaded, list):
        return [], f"{BRIEF_DIRNAME}/{FILENAME} is not a list, so no field was added"
    return loaded, None


def settle(
    work_dir: Path, doc: dict[str, Any], skill_dir: Path | None, sample_payload: Any
) -> Outcome:
    """Apply the work dir's entries to ``doc`` and record the outcome.

    ``skill_dir`` is where each quote is checked; asking to apply entries
    without one is a programming error, not a reason to skip them."""
    entries, problem = read(work_dir)
    record = work_dir / BRIEF_DIRNAME / RECORD_FILENAME
    if problem is not None:
        out = Outcome(refused=[({}, problem)])
    elif not entries:
        record.unlink(missing_ok=True)
        return Outcome()
    else:
        if skill_dir is None:
            raise ValueError(
                f"{BRIEF_DIRNAME}/{FILENAME} has entries, and their quotes need the skill dir"
            )
        out = apply(doc, entries, SkillText.read(skill_dir), sample_payload)
    record.write_text(
        json.dumps(
            {"applied": out.applied, "refused": [{"entry": e, "why": w} for e, w in out.refused]},
            indent=1,
        ),
        encoding="utf-8",
    )
    return out


def recorded(work_dir: Path) -> Outcome:
    record = work_dir / BRIEF_DIRNAME / RECORD_FILENAME
    if not record.is_file():
        return Outcome()
    doc = json.loads(record.read_text(encoding="utf-8"))
    return Outcome(
        applied=list(doc.get("applied") or []),
        refused=[(r.get("entry") or {}, str(r.get("why"))) for r in doc.get("refused") or []],
    )


def _label(entry: dict[str, Any]) -> str:
    name = entry.get("name")
    return f"`{name}`" if isinstance(name, str) and name else "an entry"


def report_questions(work_dir: Path) -> list[tuple[str, str]]:
    """Each applied and refused entry, as an Open questions ``(title, detail)``."""
    out = recorded(work_dir)
    questions = [
        (
            f"The agent added a pipeline input field: {a['name']}",
            f"`{a['name']}` ({a['type']}, {'required' if a['required'] else 'optional'}): "
            f'{a["description"]} Skill: "{a["quote"]}"',
        )
        for a in out.applied
    ]
    questions += [
        ("An input field the agent asked for was not added", f"{_label(e)}: {why}")
        for e, why in out.refused
    ]
    return questions


def refusal_note(work_dir: Path) -> str | None:
    """The refused entries, for the agent's next repair turn."""
    refused = recorded(work_dir).refused
    if not refused:
        return None
    lines = [f"- {_label(e)}: {why}" for e, why in refused]
    return (
        f"\n\nrote did not add these {BRIEF_DIRNAME}/{FILENAME} entries to the pipeline "
        "input:\n" + "\n".join(lines)
    )
