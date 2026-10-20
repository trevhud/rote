"""The compiled-for backend is recorded beside the IR, and emit honors it.

Without the record, `rote emit compiled/pipeline.yaml` after a
`--backend api` compile defaulted to mcp and silently emitted MCP calls in
place of the vendor code the compile researched. Also: python and temporal
never call a binding, so they advertise no MCP server for one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from rote.cli import _mcp_requirements, main
from rote.compiler import Compiler
from rote.compiler.backend import BACKEND_FILENAME, read_backend, write_backend
from rote.compiler.verify import EmitTarget
from rote.ir import load_pipeline
from tests.test_api_backend import CITED_MODULE, PIPELINE_YAML, _deliverable, _ResearchingDriver


def _compiled(tmp_path: Path, backend: str | None) -> Path:
    root = _deliverable(tmp_path / "compiled", CITED_MODULE)
    if backend is not None:
        write_backend(root, backend)  # type: ignore[arg-type]
    return root


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("target", "recorded"),
    [(EmitTarget("dbos", "api"), "api"), (EmitTarget("dbos", "mcp"), "mcp"), (None, None)],
)
async def test_a_compile_records_its_backend(
    tmp_path: Path, target: EmitTarget | None, recorded: str | None
) -> None:
    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: dock\n---\n\n# Dock\n", encoding="utf-8")
    rubric = tmp_path / "rubric"
    rubric.mkdir()
    (rubric / "SKILL.md").write_text("# rubric", encoding="utf-8")
    compiler = Compiler(
        system1=False,  # a fake rubric
        compiler_skill_dir=rubric,
        emit_target=target,
        run_generated_tests=False,
        tool_discovery="none",
    )
    compiler.select_driver = lambda: _ResearchingDriver(CITED_MODULE, [])  # type: ignore[method-assign]
    await compiler.compile(skill, tmp_path / "out")
    assert read_backend(tmp_path / "out") == recorded
    assert "external_backend" not in (tmp_path / "out" / "pipeline.yaml").read_text()


@pytest.mark.parametrize(
    ("recorded", "flag", "mcp_emitted", "says"),
    [
        ("api", None, False, "using --backend api, recorded when this pipeline was compiled"),
        ("api", "mcp", True, "compiled for --backend api"),
        ("mcp", "api", False, "compiled for --backend mcp"),
        (None, None, True, None),
        ("api", "api", False, None),
    ],
)
def test_emit_uses_the_recorded_backend_unless_the_flag_overrides_it(
    tmp_path: Path,
    capsys: Any,
    recorded: str | None,
    flag: str | None,
    mcp_emitted: bool,
    says: str | None,
) -> None:
    root = _compiled(tmp_path, recorded)
    out = tmp_path / "rt"
    argv = ["emit", str(root / "pipeline.yaml"), "--runtime", "dbos", "--out", str(out)]
    if flag is not None:
        argv += ["--backend", flag]
    assert main(argv) == 0
    err = capsys.readouterr().err
    assert (out / "extracted" / "_rote_mcp.py").is_file() is mcp_emitted
    if says is None:
        assert "recorded" not in err and "compiled for --backend" not in err
    else:
        assert says in err
    if flag is not None and recorded is not None and flag != recorded:
        assert err.count("warning: this pipeline was compiled for") == 1


def test_an_unreadable_record_is_an_error_not_a_default(tmp_path: Path, capsys: Any) -> None:
    root = _compiled(tmp_path, None)
    (root / BACKEND_FILENAME).write_text('{"external_backend": "grpc"}', encoding="utf-8")
    rc = main(
        ["emit", str(root / "pipeline.yaml"), "--runtime", "dbos", "--out", str(tmp_path / "rt")]
    )
    assert rc == 2
    assert "records external_backend 'grpc'" in capsys.readouterr().err


def test_compile_update_keeps_the_recorded_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    from tests.test_cli import _install_mock_compiler

    captured: list[Any] = []
    _install_mock_compiler(monkeypatch)
    original = __import__("rote.cli", fromlist=["Compiler"]).Compiler

    class _Capturing(original):  # type: ignore[misc, valid-type]
        def __init__(self, **kwargs: Any) -> None:
            captured.append(kwargs["emit_target"].external_backend)
            super().__init__(**kwargs)

    monkeypatch.setattr("rote.cli.Compiler", _Capturing)
    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: t\n---\n", encoding="utf-8")
    out = tmp_path / "out"
    (out / "compiled").mkdir(parents=True)
    write_backend(out / "compiled", "api")
    base = ["compile", str(skill), "--local", "--runtime", "dbos", "--no-eval", "--out", str(out)]
    for extra in (["--update"], ["--update", "--backend", "mcp"], []):
        assert main(base + extra) == 0
        write_backend(out / "compiled", "api")
    capsys.readouterr()
    # Recorded on update; the flag wins with a warning; a fresh compile starts at mcp.
    assert captured == ["api", "mcp", "mcp"]


@pytest.mark.parametrize(
    ("runtime", "backend", "servers"),
    [
        ("dbos", "mcp", ["gmail"]),
        ("dbos", "api", []),
        ("python", "mcp", []),
        ("temporal", "mcp", []),
        (None, "mcp", ["gmail"]),
    ],
)
def test_only_runtimes_that_call_bindings_advertise_their_servers(
    tmp_path: Path, runtime: str | None, backend: str, servers: list[str]
) -> None:
    (tmp_path / "pipeline.yaml").write_text(PIPELINE_YAML, encoding="utf-8")
    pipeline = load_pipeline(tmp_path / "pipeline.yaml")
    assert [e["server"] for e in _mcp_requirements(pipeline, backend, runtime)] == servers


@pytest.mark.parametrize(("runtime", "servers"), [("dbos", ["gmail"]), ("python", [])])
def test_emit_reports_servers_only_for_a_runtime_that_calls_them(
    tmp_path: Path, capsys: Any, runtime: str, servers: list[str]
) -> None:
    import json

    root = _compiled(tmp_path, None)
    argv = ["emit", str(root / "pipeline.yaml"), "--runtime", runtime, "--json"]
    assert main(argv + ["--out", str(tmp_path / "rt")]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert [e["server"] for e in payload["mcp_servers"]] == servers


@pytest.mark.parametrize(("runtime", "listed"), [("dbos", True), ("temporal", False)])
def test_the_compile_summary_lists_servers_only_for_a_runtime_that_calls_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any, runtime: str, listed: bool
) -> None:
    from tests.test_cli import _install_mock_compiler

    (tmp_path / "p.yaml").write_text(PIPELINE_YAML, encoding="utf-8")
    _install_mock_compiler(monkeypatch, tmp_path / "p.yaml")
    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: t\n---\n", encoding="utf-8")
    main(
        ["compile", str(skill), "--local", "--runtime", runtime, "--no-eval"]
        + ["--out", str(tmp_path / "out")]
    )
    captured = capsys.readouterr()
    assert ("required MCP servers: gmail" in captured.out + captured.err) is listed


@pytest.mark.parametrize(
    ("flag", "rejected"), [(["--backend", "mcp"], True), (["--backend", "api"], True), ([], False)]
)
def test_cloud_mode_rejects_any_explicit_backend(
    tmp_path: Path, flag: list[str], rejected: bool
) -> None:
    """An explicit --backend, even the default value, is a local-only choice:
    checking `!= "mcp"` would let `--backend mcp` through to the cloud."""
    from rote.cli import _build_parser, _reject_local_only_cloud_flags

    args = _build_parser().parse_args(
        ["compile", str(tmp_path), "--out", str(tmp_path / "o"), *flag]
    )
    message = _reject_local_only_cloud_flags(args)
    if rejected:
        assert message is not None and "--backend" in message
    else:
        assert message is None or "--backend" not in message
