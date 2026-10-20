"""The claude-cli lane's effort check, against a stub ``claude`` process.

``claude -p`` does not fail on an unknown ``--effort``: it warns on stderr
and runs at its default, exit 0 (Claude Code 2.1.304). The helper asks the
CLI about a level with a free ``claude --effort <level> --version`` call
before any paid one, and first gives it a level nothing accepts, so a CLI
release whose warning rote no longer recognizes fails loudly instead of
letting every bad level through.

The stub is a real executable (passed by absolute path; the suite's guard
blocks only bare-name ``claude`` lookups) that logs every invocation, so
"no paid call happened" is an observation, not an inference.
"""

from __future__ import annotations

import json
import stat
import sys
import textwrap
from collections.abc import Iterator
from pathlib import Path

import pytest

from rote.inference import _runtime_helper as helper

_STUB = textwrap.dedent(
    """\
    #!{python}
    import json, os, sys
    args = sys.argv[1:]
    with open(os.environ["STUB_LOG"], "a") as log:
        log.write(json.dumps(args) + "\\n")
    if "--version" in args:
        level = args[args.index("--effort") + 1] if "--effort" in args else None
        if level is not None and level not in ("low", "medium", "high", "xhigh", "max"):
            wording = os.environ.get("STUB_WARNING", "")
            if wording:
                sys.stderr.write(wording.format(level=level) + "\\n")
        print("9.9.9 (Claude Code)")
        sys.exit(0)
    sys.stdin.read()
    print(json.dumps({{
        "type": "result", "is_error": False, "num_turns": 1,
        "result": json.dumps({{"grade": 9}}), "structured_output": {{"grade": 9}},
        "usage": {{"input_tokens": 3, "output_tokens": 2}},
    }}))
    """
)

TODAY = (
    "Warning: Unknown --effort value '{level}' — ignoring it and using the default "
    "effort. Valid values: low, medium, high, xhigh, max."
)


@pytest.fixture()
def stub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    path = tmp_path / "claude"
    path.write_text(_STUB.format(python=sys.executable), encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    log = tmp_path / "calls.jsonl"
    monkeypatch.setenv("STUB_LOG", str(log))
    monkeypatch.setenv("STUB_WARNING", TODAY)
    monkeypatch.setattr(helper.shutil, "which", lambda _name: str(path))
    monkeypatch.setenv("ROTE_INFERENCE", "claude-cli")
    for var in (
        "ROTE_JUDGE_EFFORT",
        "ROTE_EFFORT_GRADE_ESSAY",
        "ROTE_EFFORT_OTHER_JUDGE",
        "ROTE_EFFORT",
    ):
        monkeypatch.delenv(var, raising=False)
    helper._CHECKED_EFFORTS.clear()
    yield log
    helper._CHECKED_EFFORTS.clear()


def _calls(log: Path) -> list[list[str]]:
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]


def _paid(log: Path) -> list[list[str]]:
    return [call for call in _calls(log) if "-p" in call]


def _judge() -> dict[str, object]:
    return helper.call_judge(
        node_id="grade_essay",
        client="anthropic",
        model="claude-sonnet-4-6",
        prompt="Grade it.",
        output_schema={"type": "object", "properties": {"grade": {"type": "integer"}}},
    )


def test_a_valid_level_is_checked_once_then_passed(
    stub: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ROTE_JUDGE_EFFORT", "low")
    assert _judge() == {"grade": 9}
    assert _judge() == {"grade": 9}
    probes = [c for c in _calls(stub) if "--version" in c]
    # The self-check and the level, once each for the process, not per call.
    assert probes == [
        ["--effort", helper._EFFORT_PROBE, "--version"],
        ["--effort", "low", "--version"],
    ]
    paid = _paid(stub)
    assert len(paid) == 2
    assert all(c[c.index("--effort") + 1] == "low" for c in paid)


def test_no_effort_means_no_check_and_no_flag(stub: Path) -> None:
    assert _judge() == {"grade": 9}
    assert _calls(stub) and all("--effort" not in c for c in _calls(stub))


@pytest.mark.parametrize(
    "wording",
    [
        TODAY,
        "error: invalid effort '{level}'",
        "Effort level '{level}' is not supported by this model",
        "warning: unrecognized value for --effort: {level}",
    ],
)
def test_an_unknown_level_fails_before_any_paid_call_whatever_the_wording(
    stub: Path, monkeypatch: pytest.MonkeyPatch, wording: str
) -> None:
    monkeypatch.setenv("STUB_WARNING", wording)
    monkeypatch.setenv("ROTE_JUDGE_EFFORT", "turbo")
    with pytest.raises(RuntimeError, match="turbo"):
        _judge()
    assert _paid(stub) == []


def test_a_cli_whose_warning_rote_cannot_read_fails_loudly(
    stub: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the CLI stops warning (or rewords past recognition), every bad
    level would pass silently. The self-check turns that into an error."""
    monkeypatch.setenv("STUB_WARNING", "")
    monkeypatch.setenv("ROTE_JUDGE_EFFORT", "low")
    with pytest.raises(RuntimeError, match="cannot verify the effort level"):
        _judge()
    assert _paid(stub) == []


def test_a_warning_about_something_else_is_not_an_effort_rejection() -> None:
    assert helper._effort_warning("Warning: unknown option --foo\n") is None
    assert helper._effort_warning("using effort high\n") is None


def test_a_probe_that_prints_nothing_is_refused_by_the_self_check_itself(
    stub: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Calls the check directly, so neither the per-level cache nor the
    paid run can be what raises: the only probe made is the self-check's,
    and its silence is the error."""
    monkeypatch.setenv("STUB_WARNING", "")
    binary = helper.shutil.which("claude")
    assert binary is not None
    with pytest.raises(RuntimeError, match="cannot verify the effort level") as excinfo:
        helper._check_cli_effort(binary, "low", "grade_essay")
    assert "printed no warning rote recognizes" in str(excinfo.value)
    assert _calls(stub) == [["--effort", helper._EFFORT_PROBE, "--version"]]
    # A failed self-check is not remembered as a pass.
    assert (binary, helper._EFFORT_PROBE) not in helper._CHECKED_EFFORTS


def test_the_checks_run_once_per_process_across_nodes_and_calls(
    stub: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Counts the stub's probe invocations: one self-check and one per
    distinct level, however many calls and nodes use them."""
    monkeypatch.setenv("ROTE_JUDGE_EFFORT", "low")
    monkeypatch.setenv("ROTE_EFFORT_OTHER_JUDGE", "high")
    for _ in range(3):
        assert _judge() == {"grade": 9}
    for _ in range(2):
        helper.call_judge(
            node_id="other_judge",
            client="anthropic",
            model="claude-sonnet-4-6",
            prompt="Grade it.",
            output_schema={"type": "object", "properties": {"grade": {"type": "integer"}}},
        )
    probes = [c[1] for c in _calls(stub) if "--version" in c]
    assert probes == [helper._EFFORT_PROBE, "low", "high"]
    assert len(_paid(stub)) == 5
