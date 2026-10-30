"""Whether a skill and its compiled form did the same thing: the field compare.

``rote eval --run --runtime skill`` (:mod:`rote.eval.skill_ab`) and ``rote
try`` both end by putting two outputs side by side, the original skill's and
the compiled one's. This module is that comparison, and the one place the
rules for it live.

How fields compare. A field names a value in each output (JSON pointers,
with a reference's ``section`` / ``measure`` narrowing, read by
:func:`rote.demo.value_at`), and values are compared after
:func:`rote.bench.fidelity_score.normalize` (whitespace and case folded,
numbers to the cent). What a difference means depends on what the field is
expected to do:

======== ==================== ============================================
expect   a difference reads   where the expectation comes from
======== ==================== ============================================
equal    ``differs``          a reference, or the IR for a field only code
                              and tools produce
drifts   ``drifted``          a reference: the value is live data
wording  ``differs-judge``    the IR: an llm_judge or agent_loop node is
                              upstream, so the words can change
======== ==================== ============================================

A value missing from either output is ``differs`` whatever the field
expects. "Same result" rests only on ``equal`` fields, and is never claimed
from zero of them: a comparison with no ``equal`` field says so instead
(:attr:`Comparison.headline`).

It also holds the one ``rote compile`` argv a measurement compiles with
(:func:`compile_command`), pinned so the user's config and login can never
change what was measured.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from rote import demo

if TYPE_CHECKING:
    from rote.ir import Pipeline

Expect = Literal["equal", "drifts", "wording"]
Status = Literal["matched", "differs", "drifted", "differs-judge"]

_ON_MISMATCH: dict[Expect, Status] = {
    "equal": "differs",
    "drifts": "drifted",
    "wording": "differs-judge",
}

_MODEL_KINDS = ("llm_judge", "agent_loop")


# ───────── Fields ─────────


@dataclass(frozen=True)
class FieldSpec:
    """One value both outputs are compared on."""

    name: str
    #: JSON pointer into the original skill's output.
    original: str
    #: JSON pointer into the compiled side's output.
    compiled: str
    section: str | None
    measure: demo.Measure | None
    expect: Expect
    why: str

    @classmethod
    def from_reference(cls, ref_field: demo.Field, *, same_shape: bool = False) -> FieldSpec:
        """A reference field. ``same_shape``: both outputs are the skill's
        shape (an agent wrote each one to the same contract), so the skill
        pointer reads both."""
        return cls(
            name=ref_field.name,
            original=ref_field.skill,
            compiled=ref_field.skill if same_shape else ref_field.pipeline,
            section=ref_field.section,
            measure=ref_field.measure,
            expect=ref_field.expect,
            why=ref_field.why,
        )


def pointer_token(key: str) -> str:
    return key.replace("~", "~0").replace("/", "~1")


def model_nodes(pipeline: Pipeline) -> list[str]:
    """The nodes that call a model when the pipeline runs."""
    return [n.id for n in pipeline.nodes if n.kind.value in _MODEL_KINDS]


def _upstream(pipeline: Pipeline, node_id: str) -> set[str]:
    """``node_id`` and every node whose output can reach it (edges and ``inputs:``)."""
    from rote.ir import parse_input_ref

    parents: dict[str, set[str]] = {n.id: set() for n in pipeline.nodes}
    for edge in pipeline.edges:
        parents.setdefault(edge.to, set()).add(edge.from_)
    for node in pipeline.nodes:
        for ref in (node.inputs or {}).values():
            source = parse_input_ref(ref).node_id
            if source is not None:
                parents[node.id].add(source)
    seen: set[str] = set()
    stack = [node_id]
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        stack.extend(parents.get(current, ()))
    return seen


def _expectation(pipeline: Pipeline, node_ids: Sequence[str]) -> tuple[Expect, str]:
    judged = sorted(
        set().union(*(_upstream(pipeline, n) for n in node_ids)) & set(model_nodes(pipeline))
    )
    if judged:
        return "wording", f"produced downstream of {', '.join(judged)}, which call a model"
    return "equal", "produced by code and tool calls only"


def fields_from_ir(pipeline: Pipeline) -> tuple[FieldSpec, ...]:
    """The fields to compare a skill's output with the pipeline's own result:
    every declared key of every exit node (the pipeline returns ``{exit_id:
    result}``; the skill's result.json is read at the same key, unprefixed).

    A field is ``wording`` when an llm_judge or agent_loop node is the exit
    node or upstream of it, and ``equal`` otherwise: code and tool calls
    alone produce it, so the same input should give the same value.
    """
    single_exit = len(pipeline.exit_nodes) == 1
    fields: list[FieldSpec] = []
    for exit_id in pipeline.exit_nodes:
        node = pipeline.node_by_id(exit_id)
        expect, why = _expectation(pipeline, [exit_id])
        keys: list[str] = []
        if isinstance(node.output, dict) and node.output:
            keys = list(node.output)
        elif isinstance(node.output_schema, dict) and node.output_schema.get("type") == "object":
            keys = list((node.output_schema.get("properties") or {}).keys())
        if not keys:
            token = f"/{pointer_token(exit_id)}"
            fields.append(FieldSpec(exit_id, token, token, None, None, expect, why))
            continue
        for key in keys:
            fields.append(
                FieldSpec(
                    name=key if single_exit else f"{exit_id}.{key}",
                    original=f"/{pointer_token(key)}",
                    compiled=f"/{pointer_token(exit_id)}/{pointer_token(key)}",
                    section=None,
                    measure=None,
                    expect=expect,
                    why=why,
                )
            )
    return tuple(fields)


def contract_fields(pipeline: Pipeline, names: Sequence[str]) -> tuple[FieldSpec, ...]:
    """Fields named by an output contract both sides wrote to (a bench case's
    ``output_fields``): the same top-level key in each output.

    The contract does not say which node produces a field, so a pipeline with
    any model call makes every field ``wording``; one without makes them
    ``equal``. That is conservative: it never claims a same result a model
    could have changed.
    """
    expect, why = _expectation(pipeline, list(pipeline.exit_nodes))
    if expect == "wording":
        why = f"the pipeline calls a model ({', '.join(model_nodes(pipeline))}) somewhere"
    return tuple(
        FieldSpec(
            name, f"/{pointer_token(name)}", f"/{pointer_token(name)}", None, None, expect, why
        )
        for name in names
    )


@dataclass(frozen=True)
class FieldResult:
    spec: FieldSpec
    status: Status
    #: The compared values, after section and measure (None when missing).
    original_value: Any
    compiled_value: Any
    #: Why a value could not be read, when one could not.
    problem: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.spec.name,
            "expect": self.spec.expect,
            "status": self.status,
            "why": self.spec.why,
            "original": self.original_value,
            "compiled": self.compiled_value,
            "problem": self.problem,
        }


def canonical(value: Any) -> Any:
    """``normalize`` applied to every leaf, order kept (a reading list's
    order is part of it)."""
    from rote.bench.fidelity_score import normalize

    if isinstance(value, Mapping):
        return {str(k): canonical(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [canonical(v) for v in value]
    return normalize(value)


def compare_outputs(
    fields: Sequence[FieldSpec], original_output: Any, compiled_output: Any
) -> tuple[FieldResult, ...]:
    results: list[FieldResult] = []
    for spec in fields:
        problems: list[str] = []
        values: list[Any] = []
        for side, document, pointer in (
            ("original", original_output, spec.original),
            ("compiled", compiled_output, spec.compiled),
        ):
            try:
                values.append(demo.value_at(document, pointer, spec.section, spec.measure))
            except demo.DemoError as e:
                values.append(None)
                problems.append(f"{side} output: {e}")
        if problems:
            status: Status = "differs"
        elif canonical(values[0]) == canonical(values[1]):
            status = "matched"
        else:
            status = _ON_MISMATCH[spec.expect]
        results.append(FieldResult(spec, status, values[0], values[1], "; ".join(problems) or None))
    return tuple(results)


@dataclass(frozen=True)
class Comparison:
    fields: tuple[FieldResult, ...]

    @property
    def compared(self) -> int:
        """The ``equal`` fields: the only ones a "same result" claim rests on."""
        return sum(1 for f in self.fields if f.spec.expect == "equal")

    @property
    def reproduced(self) -> int:
        return sum(1 for f in self.fields if f.spec.expect == "equal" and f.status == "matched")

    @property
    def ok(self) -> bool:
        """No field differs where it must not, and at least one was compared
        exactly: zero ``equal`` fields prove nothing either way."""
        return self.compared > 0 and all(f.status != "differs" for f in self.fields)

    @property
    def same_result(self) -> bool:
        return self.ok and self.reproduced == self.compared

    @property
    def headline(self) -> str:
        if self.compared == 0:
            return "no fields to compare exactly"
        return f"reproduced {self.reproduced} of {self.compared} fields"

    def to_dict(self) -> dict[str, Any]:
        return {
            "headline": self.headline,
            "reproduced": self.reproduced,
            "compared": self.compared,
            "same_result": self.same_result,
            "fields": [f.to_dict() for f in self.fields],
        }


def verdict(comparison: Comparison) -> str:
    """One sentence a person reads under the field table."""
    cmp = comparison
    if cmp.compared == 0:
        return "No field is expected to match exactly, so no same-result claim is made."
    if cmp.same_result:
        return f"Same result on all {cmp.compared} fields expected to match exactly."
    if cmp.reproduced == cmp.compared:
        return "Not the same result: a compared value is missing from one of the outputs."
    return (
        f"Not the same result: {cmp.compared - cmp.reproduced} of {cmp.compared} "
        "fields expected to match exactly did not."
    )


# ───────── The compile a measurement runs ─────────


@dataclass(frozen=True)
class PinnedSetting:
    """A layered compile setting the measurement pins, and why."""

    flag: str
    value: str
    why: str


def pinned_settings() -> tuple[PinnedSetting, ...]:
    """The layered settings (flag > env > rote.yaml > user config > default)
    a measured compile pins on its argv, so nothing ambient changes them.

    The decider is pinned to ``rules``: ``auto`` takes whichever decision
    model's credentials happen to be in the environment, so two people
    running the same measurement would compile differently. The structure
    pass and the tie-break are pinned to their built-in defaults."""
    from rote.compiler.system1.structure import DEFAULT as DEFAULT_STRUCTURE
    from rote.compiler.system1.tiebreak import DEFAULT_TIEBREAK

    return (
        PinnedSetting("--decider", "rules", "auto depends on which credentials are present"),
        PinnedSetting("--structure", DEFAULT_STRUCTURE, "the built-in default"),
        PinnedSetting("--tiebreak", DEFAULT_TIEBREAK, "the built-in default"),
    )


def compile_command(
    skill_dir: Path,
    input_path: Path,
    out_dir: Path,
    *,
    runtime: str,
    model: str | None = None,
    agent: str = "claude",
) -> list[str]:
    """The ``rote compile`` argv (without ``rote``) a measurement compiles with.

    Every choice that rote would otherwise resolve from the user's setup is
    on the argv: ``--local`` (a logged-in user's default compiles in the
    cloud), ``--agent`` (claude unless the measurement runs on codex), the
    runtime, ``--no-eval``, and the
    :func:`pinned_settings`. The caller prints and records it, so what was
    measured can be rerun exactly."""
    argv = [
        "compile",
        str(skill_dir),
        "--sample-input",
        str(input_path),
        "--runtime",
        runtime,
        "--out",
        str(out_dir),
        "--agent",
        agent,
        "--local",
        "--no-eval",
    ]
    if model is not None:
        argv += ["--model", model]
    for setting in pinned_settings():
        argv += [setting.flag, setting.value]
    return argv
