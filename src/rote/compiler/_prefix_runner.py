"""Run a deliverable's deterministic prefix on a sample input. Stdlib only.

rote copies this file's source into a scratch dir and runs it with
``python -S`` inside the OS sandbox the generated tests use, with the
deliverable dir on ``PYTHONPATH``. It imports nothing from rote: the plan
it executes (``plan.json``, built by :mod:`rote.compiler.prefix`) already
says, per node, which function to call and where each argument comes
from. That keeps the semantics in one place (rote's plan builder, which
mirrors the emitted step bodies) and this file a dumb interpreter.

Usage: ``python -S _prefix_runner.py <plan.json> <report.json>``. The
report file is opened, and its path dropped from ``sys.argv``, before any
agent module is imported, and the report is written through that handle.
Never stdout: the agent's code runs in this process and could print a
forged report and ``os._exit(0)``. A run that exits before writing leaves
the file empty, which rote reports as a crash. (The agent's code shares
this process, so a deliberate attack on this file is the same trust
boundary as pytest's exit code; the sandbox, not this, is the boundary.)
"""

from __future__ import annotations

import importlib
import inspect
import json
import re
import sys
import traceback
from pathlib import Path
from typing import Any

_MISSING = object()


def _serialize(obj: Any) -> Any:
    """The emitted ``_serialize``: pydantic models and tuples to plain values."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump(by_alias=True)
    if isinstance(obj, (list, tuple)):
        return [_serialize(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    return obj


def _not_json(value: Any, path: str = "result") -> str | None:
    """Where ``value`` stops being plain JSON, or None when it is JSON throughout.

    A step result crosses a durable boundary (a DBOS row, a Workers step, a
    JSON run result), so a set, a datetime or an object there either fails
    or silently turns into its ``repr``: a real compile's duplicate check
    returned ``{'EXP-2412'}``, a Python set, and the run reported the string
    instead of the list."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return None
    if isinstance(value, list):
        for i, item in enumerate(value):
            where = _not_json(item, f"{path}[{i}]")
            if where:
                return where
        return None
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                return f"{path} has a {type(key).__name__} key {key!r}"
            where = _not_json(item, f"{path}.{key}")
            if where:
                return where
        return None
    return f"{path} is a {type(value).__name__}"


class BindingError(Exception):
    """An ``inputs:`` reference that the upstream result cannot satisfy."""


def _keys(value: Any) -> str:
    if isinstance(value, dict):
        return "keys " + (", ".join(sorted(map(str, value))) or "none")
    return f"a {type(value).__name__}, not an object"


def _resolve(binding: list[Any], sample: dict[str, Any], results: dict[str, Any]) -> Any:
    """One payload value, or ``_MISSING`` for an omitted optional input field."""
    param, source, node_id, fld = binding
    if source == "input":
        return sample
    if source == "input_field":
        return sample.get(fld, _MISSING)
    upstream = results[node_id]
    if fld is None:
        return upstream
    if not isinstance(upstream, dict) or fld not in upstream:
        raise BindingError(
            f"`{param}` is bound to {node_id}.output.{fld}, but {node_id} returned "
            f"{_keys(upstream)}"
        )
    return upstream[fld]


def _location(exc: BaseException, root: Path) -> dict[str, Any] | None:
    """The innermost traceback frame in the deliverable: file, line, code."""
    found = None
    for frame in traceback.extract_tb(exc.__traceback__):
        try:
            rel = Path(frame.filename).resolve().relative_to(root)
        except ValueError:
            continue
        found = {"path": str(rel), "line": frame.lineno, "code": (frame.line or "").strip()}
    return found


def _call(fn: Any, payload: dict[str, Any]) -> Any:
    result = fn(**payload)
    if inspect.isawaitable(result):
        import asyncio

        async def _await() -> Any:
            return await result

        result = asyncio.run(_await())
    return result


def _mcp_arguments(step: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """The arguments this step sends, built the way every emitter builds them."""
    if step["args"] is None:
        return payload
    arguments = {}
    for tool_arg, origin, value in step["args"]:
        arguments[tool_arg] = value if origin in ("constant", "literal") else payload[value]
    return arguments


def _mcp_result(step: dict[str, Any], arguments: dict[str, Any]) -> Any:
    """The recorded result of the call with ``arguments``, or ``_MISSING``."""
    for observed in step["observed"]:
        if observed["input"] == arguments:
            return observed["output"]
    return _MISSING


#: Probe values by shape: what the finding calls it, and a fresh empty value.
_SHAPES: dict[str, tuple[str, Any]] = {
    "list": ("a JSON list", list),
    "text": ("text", str),
    "object": ("an object", dict),
}
#: The errors that mean code treated a list or a string as a dict. Any other
#: AttributeError or TypeError is about something else and is no verdict.
_DICT_ASSUMPTION = re.compile(
    r"'(?:list|str)' object has no attribute '(?:get|keys|items|values|setdefault|pop)'"
    r"|list indices must be integers or slices, not str"
    r"|string indices must be integers"
)


def _probe_shapes(
    step: dict[str, Any],
    sample: dict[str, Any],
    results: dict[str, Any],
    unrecorded: dict[str, list[str]],
    root: Path,
) -> dict[str, Any] | None:
    """Run a code step that reads an unrecorded MCP result whole, on a list and on text.

    rote knows a tool's input schema, never its result's shape, and MCP
    tools commonly return a bare JSON list (a real Gmail search_threads
    did). A consumer that only works on a dict crashes on every real run,
    and nothing before it would say so: 7 of 9 real ops-report compiles
    called ``.get("threads")`` on that list. A tool that answers in text
    (Drive's read_file_content returns CSV) reaches the code as a string, and
    the next compile called ``.get`` on that. The probe calls the step once
    with an empty list and once with an empty string in place of each
    unrecorded result. Only a crash that assumes a dict counts
    (AttributeError, TypeError). Any other outcome says nothing about the
    real data, so the step stays not reached and its result is never used
    downstream.
    """
    if step["kind"] != "call" or step["fan_param"] is not None:
        return None
    probed = [b for b in step["bindings"] if b[1] == "node" and b[2] in unrecorded and b[3] is None]
    others = [b for b in step["bindings"] if b not in probed]
    if not probed:
        return None
    # An other input whose node did not run cannot resolve (KeyError below):
    # such a consumer is never called, so it gets no verdict.
    try:
        payload = {}
        for binding in others:
            value = _resolve(binding, sample, results)
            if value is not _MISSING:
                payload[binding[0]] = value
        fn = getattr(importlib.import_module(step["module"]), step["symbol"])
    except (Exception, SystemExit):  # noqa: BLE001 (no verdict on made-up data)
        return None
    shapes = [s for s in _SHAPES if any(s in unrecorded[b[2]] for b in probed)]
    for shape in shapes:
        label, empty = _SHAPES[shape]
        try:
            _call(fn, payload | {b[0]: empty() for b in probed} | (step["constants"] or {}))
        except (AttributeError, TypeError) as e:
            if not _DICT_ASSUMPTION.search(str(e)):
                continue
            return {
                "status": "shape_assumption",
                "upstream": sorted({b[2] for b in probed}),
                "shape": label,
                "type": type(e).__name__,
                "message": str(e)[:300],
                "at": _location(e, root),
            }
        except (Exception, SystemExit):  # noqa: BLE001 (no verdict on made-up data)
            continue
    return None


def run(plan: dict[str, Any], root: Path) -> dict[str, Any]:
    sample = plan["sample"]
    results: dict[str, Any] = {}
    report: dict[str, Any] = {}
    #: MCP nodes that ran with no recorded result, and the shapes to probe.
    unrecorded: dict[str, list[str]] = {}
    for step in plan["steps"]:
        node_id = step["id"]
        if step["kind"] == "skip":
            report[node_id] = {"status": "not_deterministic", "reason": step["reason"]}
            continue
        not_run = [b[2] for b in step["bindings"] if b[1] == "node" and b[2] not in results]
        if not_run:
            report[node_id] = _probe_shapes(step, sample, results, unrecorded, root) or {
                "status": "not_reached",
                "upstream": sorted(set(not_run)),
            }
            continue
        try:
            payload = {}
            for binding in step["bindings"]:
                value = _resolve(binding, sample, results)
                if value is not _MISSING:
                    payload[binding[0]] = value
            if step["kind"] == "mcp":
                # The arguments are reported either way: rote checks them
                # against the tool's schema, recorded call or not.
                arguments = _mcp_arguments(step, payload)
                output = _mcp_result(step, arguments)
                if output is _MISSING:
                    report[node_id] = {"status": "no_recording", "arguments": arguments}
                    unrecorded[node_id] = step.get("probe_shapes") or ["list", "text"]
                    continue
                results[node_id] = output
                report[node_id] = {"status": "ok", "recorded": True, "arguments": arguments}
                continue
            module = importlib.import_module(step["module"])
            fn = getattr(module, step["symbol"])
            payload = {**payload, **(step["constants"] or {})}
            fan = step["fan_param"]
            if fan is None:
                output = _serialize(_call(fn, payload))
            else:
                items = payload.get(fan)
                if not isinstance(items, list):
                    raise BindingError(
                        f"fan_out over `{fan}` expected a list, got {type(items).__name__}"
                    )
                output = [_serialize(_call(fn, {**payload, fan: item})) for item in items]
            where = _not_json(output)
            if where:
                report[node_id] = {"status": "not_json", "message": where}
                continue
            results[node_id] = output
            report[node_id] = {"status": "ok"}
        except BindingError as e:
            report[node_id] = {"status": "binding_error", "message": str(e)}
        except NotImplementedError as e:
            report[node_id] = {"status": "stub", "message": str(e)[:300]}
        except (Exception, SystemExit) as e:  # noqa: BLE001 (each one is the agent's to fix)
            report[node_id] = {
                "status": "error",
                "type": type(e).__name__,
                "message": str(e)[:500],
                "at": _location(e, root),
            }
    return report


def main(argv: list[str]) -> int:
    plan = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    # Opened before any agent module is imported (truncating it), and the
    # path is dropped from argv so agent code is not handed it.
    out = open(argv[2], "w", encoding="utf-8")  # noqa: SIM115 (held across agent code)
    del sys.argv[2:]
    root = Path(plan["root"]).resolve()
    sys.path.insert(0, str(root))
    report = run(plan, root)
    out.write(json.dumps(report, default=str))
    out.close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
