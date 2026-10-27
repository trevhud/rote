"""The one rule that renders a gate's message, in Python and in its TS twin.

``notify.message_template`` is filled from the gate's bound inputs by the
emitted ``_render_gate_message`` (Python runtimes) and ``renderGateMessage``
(TS runtimes). The rule: a string value is used as is, any other value is
written as JavaScript's ``JSON.stringify`` writes it (so Python's ``2.0``,
``1e+16`` and key order cannot make the same run read differently per
runtime), a placeholder whose value the run does not have (an omitted
optional input, a key an upstream result lacks) is kept as written,
and every invisible character (controls, format characters such as bidi
overrides, line and paragraph separators, lone surrogates) is spelled
``\\uXXXX`` so a value can never drive the terminal it is shown on. The
newline and the tab stay, so a message keeps its lines.

The Python half is checked here against exact strings. The TS half runs
the same corpus under real node and must agree case for case (slow).
Also here: the run-side half, which turns a reported gate into the
announcement ``rote run`` prints, and the log marker parser.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from rote.adapters._py_common import GATE_MESSAGE_HELPER_PY, render_gate_message
from rote.adapters._ts_common import GATE_MESSAGE_HELPER_TS
from rote.mcp.parked import (
    GATE_PARK_MARKER,
    AuthPark,
    AuthParkWatcher,
    GateReached,
    gates_from_status,
    parse_gate_park_markers,
    wait_for_exit,
)

#: (template, values, rendered). Values go through JSON before TS sees
#: them, so every case here is one both languages can be handed.
CASES: list[tuple[str, dict[str, Any], str]] = [
    ("plain {a}", {"a": "x"}, "plain x"),
    ("{a}", {"a": 3}, "3"),
    ("{a}", {"a": -7}, "-7"),
    ("{a}", {"a": 3.0}, "3"),
    ("{a}", {"a": -0.0}, "0"),
    ("{a}", {"a": 0.1}, "0.1"),
    ("{a}", {"a": 123.456}, "123.456"),
    ("{a}", {"a": 1e16}, "10000000000000000"),
    ("{a}", {"a": 1e20}, "100000000000000000000"),
    ("{a}", {"a": 1e21}, "1e+21"),
    ("{a}", {"a": 1.5e300}, "1.5e+300"),
    ("{a}", {"a": 1e-6}, "0.000001"),
    ("{a}", {"a": 1.25e-7}, "1.25e-7"),
    ("{a}", {"a": -2.5e-10}, "-2.5e-10"),
    ("{a}", {"a": 9007199254740993}, "9007199254740992"),
    ("{a}", {"a": 12345678901234567891}, "12345678901234567000"),
    ("{a}", {"a": True}, "true"),
    ("{a}", {"a": False}, "false"),
    ("{a}", {"a": None}, "null"),
    ("{a}", {"a": ""}, ""),
    ("{a}", {"a": [1, "x", None, 2.0]}, '[1,"x",null,2]'),
    ("{a}", {"a": []}, "[]"),
    ("{a}", {"a": {}}, "{}"),
    (
        "{a}",
        {"a": {"z": 1, "10": 2, "2": 3, "01": 4, "-1": 5}},
        '{"2":3,"10":2,"z":1,"01":4,"-1":5}',
    ),
    ("{a}", {"a": {"4294967295": 1, "4294967294": 2}}, '{"4294967294":2,"4294967295":1}'),
    ("{a}", {"a": {"k": {"y": [True], "x": 1.0}}}, '{"k":{"y":[true],"x":1}}'),
    ("{a}", {"a": ['q"\\\n\x01\x7f ']}, '["q\\"\\\\\\n\\u0001\\u007f\\u2028"]'),
    ("{a}", {"a": 'q"\\\n\x01'}, 'q"\\\n\\u0001'),
    ("{a} {b}", {"a": "x"}, "x {b}"),
    ("{a}{a}", {"a": 1}, "11"),
    ("{ a } {1x} {{a}} {a-b}", {"a": "v"}, "{ a } {1x} {v} {a-b}"),
    ("{a}", {"a": "{b}"}, "{b}"),
    ("tab\there\nnext", {}, "tab\there\nnext"),
    ("cr\r{a}", {"a": "\x1b[31mred"}, "cr\\u000d\\u001b[31mred"),
    ("‮{a}", {"a": "​z﻿"}, "\\u202e\\u200bz\\ufeff"),
    ("{a}", {"a": "\U000e0001tag"}, "\\U000e0001tag"),
    ("{a}", {"a": "\U0001f600 café"}, "\U0001f600 café"),
    ("{a}", {"a": "\ud800lone"}, "\\ud800lone"),
    ("{a}", {"a": ["\udfff"]}, '["\\udfff"]'),
]


@pytest.mark.parametrize(("template", "values", "expected"), CASES)
def test_the_python_rule(template: str, values: dict[str, Any], expected: str) -> None:
    assert render_gate_message(template, values) == expected


def test_values_only_python_holds_render_as_javascript_would_have_them() -> None:
    """A Python value has no TS twin when JSON cannot carry it: NaN and the
    infinities are what JSON.stringify writes as null, a tuple is a list."""
    values = {"a": float("nan"), "b": float("inf"), "c": (1, 2)}
    assert render_gate_message("{a} {b} {c}", values) == "null null [1,2]"


def test_a_value_that_is_not_json_fails_naming_its_type() -> None:
    """Never ``str()`` of an arbitrary object: the TS runtimes could not show it."""
    with pytest.raises(TypeError, match="set is not a JSON value"):
        render_gate_message("{a}", {"a": {1, 2}})
    with pytest.raises(TypeError, match="object key 1 is not a string"):
        render_gate_message("{a}", {"a": {1: "x"}})


_TS_MAIN = """
import { readFileSync } from "node:fs";

const cases = JSON.parse(readFileSync(0, "utf8")) as [string, Record<string, unknown>][];
process.stdout.write(JSON.stringify(cases.map(([t, v]) => renderGateMessage(t, v))));
"""


@pytest.mark.slow
def test_the_ts_twin_renders_every_case_the_same(tmp_path: Path) -> None:
    node = shutil.which("node")
    assert node, "node is required to run the TS twin"
    script = tmp_path / "gate_message.mts"
    script.write_text(GATE_MESSAGE_HELPER_TS + _TS_MAIN, encoding="utf-8")
    done = subprocess.run(
        [node, str(script)],
        input=json.dumps([[t, v] for t, v, _ in CASES]),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert done.returncode == 0, done.stderr
    theirs = json.loads(done.stdout)
    differ = [
        (t, v, want, got) for (t, v, want), got in zip(CASES, theirs, strict=True) if got != want
    ]
    assert not differ, differ


#: (source, field, the value the message gets; None means the run does not
#: have it, so the placeholder is kept). A result that is not an object has
#: no field, on both sides: TS would otherwise throw on null.
FIELD_CASES: list[tuple[Any, str, Any]] = [
    ({"a": 1}, "a", 1),
    ({"a": None}, "a", "null"),
    ({"a": 1}, "b", None),
    (None, "a", None),
    ("text", "length", None),
    ([1, 2], "0", None),
    (3, "a", None),
]


def _python_field(source: Any, field: str) -> Any:
    namespace: dict[str, Any] = {}
    exec(GATE_MESSAGE_HELPER_PY, namespace)
    got = namespace["_gate_input"]("v", source, field)
    if not got:
        return None
    return "null" if got["v"] is None else got["v"]


@pytest.mark.parametrize(("source", "field", "expected"), FIELD_CASES)
def test_a_value_the_run_does_not_have_is_absent_in_python(
    source: Any, field: str, expected: Any
) -> None:
    assert _python_field(source, field) == expected


_TS_FIELD_MAIN = """
import { readFileSync } from "node:fs";

const cases = JSON.parse(readFileSync(0, "utf8")) as [unknown, string][];
process.stdout.write(
    JSON.stringify(
        cases.map(([s, f]) => {
            const v = roteGateField(s, f);
            return v === undefined ? null : v === null ? "null" : v;
        }),
    ),
);
"""


@pytest.mark.slow
def test_the_ts_twin_has_the_same_values(tmp_path: Path) -> None:
    node = shutil.which("node")
    assert node, "node is required to run the TS twin"
    script = tmp_path / "gate_field.mts"
    script.write_text(GATE_MESSAGE_HELPER_TS + _TS_FIELD_MAIN, encoding="utf-8")
    done = subprocess.run(
        [node, str(script)],
        input=json.dumps([[s, f] for s, f, _ in FIELD_CASES]),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout) == [want for _, _, want in FIELD_CASES]


# ───────── the run side ─────────


class _Channel:
    external_release = False
    remote_fix = None

    def __init__(self) -> None:
        self.gates: list[GateReached] = []

    def poll(self) -> list[AuthPark]:
        return []

    def poll_gates(self) -> list[GateReached]:
        return list(self.gates)

    def release(self, park: AuthPark) -> None:
        raise AssertionError("no auth park here")

    def close(self) -> None:
        pass


def _watcher(channel: _Channel, lines: list[str]) -> AuthParkWatcher:
    return AuthParkWatcher(channel=channel, interactive=False, echo=lines.append)


def test_each_gate_is_announced_once_with_its_message_indented() -> None:
    channel = _Channel()
    lines: list[str] = []
    watcher = _watcher(channel, lines)
    watcher.tick()
    assert lines == []
    channel.gates.append(
        GateReached("review", "approved", "Approve 3 rows?\n\nSee below.\n\n", "k1")
    )
    watcher.tick()
    watcher.tick()
    channel.gates.append(GateReached("sign_off", "signed", None, "k2"))
    channel.gates.append(GateReached("blank", "b", "\n", "k3"))
    watcher.drain()
    assert lines == [
        "rote run: waiting at gate 'review' for signal 'approved':\n"
        "  Approve 3 rows?\n"
        "\n"
        "  See below.",
        "rote run: waiting at gate 'sign_off' for signal 'signed'",
        "rote run: waiting at gate 'blank' for signal 'b'",
    ]


def test_wait_for_exit_announces_a_gate_the_run_passed_between_ticks() -> None:
    """A gate whose payload was already waiting can be reached and passed,
    and the app gone, before the watcher ticks once. The DBOS event keeps
    the gate, so the drain after exit still announces it."""
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    channel = _Channel()
    channel.gates.append(GateReached("approve", "approval", "Ship it?", "wf#0"))
    lines: list[str] = []
    wait_for_exit(proc, deadline=time.monotonic() + 5, watcher=_watcher(channel, lines))
    assert lines == ["rote run: waiting at gate 'approve' for signal 'approval':\n  Ship it?"]


def test_the_announcement_escapes_what_an_app_reports() -> None:
    """The emitted renderer escapes, but the runner prints what any app says,
    so it escapes again. On a rendered message that changes nothing."""
    channel = _Channel()
    lines: list[str] = []
    watcher = _watcher(channel, lines)
    channel.gates.append(GateReached("g", "s", "a\x1b[2Jb \\u001b kept", "k"))
    watcher.tick()
    assert lines == ["rote run: waiting at gate 'g' for signal 's':\n  a\\u001b[2Jb \\u001b kept"]


def test_gates_from_status_reads_the_reached_list_in_order() -> None:
    status = {
        "reached": [
            {"gate": "a", "signal": "sa", "message": "m"},
            {"gate": "b", "signal": "sb", "message": None},
        ]
    }
    assert gates_from_status(status, "wf") == [
        GateReached("a", "sa", "m", "wf#0"),
        GateReached("b", "sb", None, "wf#1"),
    ]
    assert gates_from_status(None, "wf") == []


def test_gate_markers_parse_and_replays_share_a_key() -> None:
    doc = {"gate": "approve", "signal": "approval", "message": "line1\nline2", "wait": "approve"}
    line = f"{GATE_PARK_MARKER} {json.dumps(doc)}"
    text = "\n".join(
        [
            "boot",
            f"[wrangler:info] {line}",
            line,  # a replay prints it again
            f"{GATE_PARK_MARKER} {{not json",
            f"{GATE_PARK_MARKER} " + json.dumps({"gate": "x", "signal": "s", "message": None}),
        ]
    )
    assert parse_gate_park_markers(text) == [
        GateReached("approve", "approval", "line1\nline2", "approve"),
        GateReached("approve", "approval", "line1\nline2", "approve"),
    ]
