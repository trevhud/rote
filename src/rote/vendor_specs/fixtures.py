"""HTTP fixtures: what a vendor API answers, per operation, for one fidelity case.

The shareable unit is one :class:`FixtureSet` per vendor API version, keyed
like the spec it answers for (``google/gmail/v1``), so a shared store can
hold specs and fixtures side by side::

    {"format_version": 1, "spec": "google/gmail/v1", "trust": "spec",
     "operations": {"gmail.users.threads.list": [
         {"when_contains": {"q": "dock-activity"},
          "body": {"threads": [{"id": "t-901", "snippet": "..."}]},
          "reads": "gmail.search_threads[0]"}]}}

A request takes the first response whose ``when`` parameters all equal its
own (``when_contains``: case-insensitive substring). ``body`` is the JSON
the API returns, or the file's text for a media download. ``reads`` names
the MCP fixture response that serves the same data (``<server>.<tool>[i]``),
so a read through the API and a read through the MCP tool count as the same
read when the bench compares a pipeline with the skill.

``trust`` says where a fixture's shape comes from: ``spec`` (validated
against the vendor's machine-readable spec at load) or ``docs`` (written
from documentation pages where no spec exists; reported as lower-trust).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from rote.vendor_specs import schema
from rote.vendor_specs.check import _value_problem
from rote.vendor_specs.model import ApiSpec

FORMAT_VERSION = 1


class FixtureResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    when: dict[str, Any] = Field(default_factory=dict)
    when_contains: dict[str, str] = Field(default_factory=dict)
    status: int = 200
    body: Any = None
    reads: str | None = None

    def matches(self, params: dict[str, Any]) -> bool:
        for key, want in self.when.items():
            if str(params.get(key)) != str(want):
                return False
        for key, want in self.when_contains.items():
            got = params.get(key)
            if not isinstance(got, str) or want.casefold() not in got.casefold():
                return False
        return True


class FixtureSet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    format_version: int = FORMAT_VERSION
    #: The spec key this set answers for: ``<vendor>/<api>/<version>``.
    spec: str
    trust: Literal["spec", "docs"] = "spec"
    #: For ``trust: docs``: the pages the fixture was written from.
    sources: list[str] = Field(default_factory=list)
    operations: dict[str, list[FixtureResponse]] = Field(default_factory=dict)


class FixtureError(ValueError):
    """A fixture that does not fit the spec it claims to answer for."""


def validate(fixtures: FixtureSet, spec: ApiSpec) -> None:
    """Hold every fixture to the spec: a known operation, and a body of the right shape."""
    if fixtures.spec != spec.key:
        raise FixtureError(f"fixture set for {fixtures.spec} checked against {spec.key}")
    errors: list[str] = []
    for op_id, responses in fixtures.operations.items():
        op = spec.operation(op_id)
        if op is None:
            errors.append(f"{op_id}: not an operation of {spec.key} rev {spec.revision}")
            continue
        declared = {**spec.global_params, **op.params}
        for i, response in enumerate(responses):
            unknown = set(response.when) | set(response.when_contains)
            unknown -= set(declared)
            if unknown:
                errors.append(f"{op_id}[{i}]: matches on undeclared parameters {sorted(unknown)}")
            # A value no request could send never matches, and the case
            # silently gets the default answer (`format: metdata`).
            for name, value in response.when.items():
                if name in declared and (problem := _value_problem(declared[name], value)):
                    errors.append(f"{op_id}[{i}]: when {name!r}: {problem}")
            for name in response.when_contains:
                if name in declared and declared[name].type != "string":
                    errors.append(
                        f"{op_id}[{i}]: when_contains {name!r}: a substring match needs a "
                        f"string parameter, and it is {declared[name].type}"
                    )
            if response.status >= 400:
                continue
            if op.media_download or (op.supports_media and response.when.get("alt") == "media"):
                if not isinstance(response.body, str):
                    errors.append(f"{op_id}[{i}]: a media download's body is the file's text")
                continue
            errors += [
                f"{op_id}[{i}] {p}"
                for p in schema.problems(spec, op.response_schema, response.body)
            ]
    if errors:
        raise FixtureError(
            f"fixtures for {spec.key} do not fit rev {spec.revision}:\n  " + "\n  ".join(errors)
        )


__all__ = ["FORMAT_VERSION", "FixtureError", "FixtureResponse", "FixtureSet", "validate"]
