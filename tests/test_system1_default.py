"""System-1 is the default compile path; ``--no-system1`` and ``--update`` opt out.

What is pinned: a flagless compile and a flagless analyze run System-1, the
opt-out runs the full rubric, ``--update`` runs the full rubric and says so,
a model decider never runs silently as rules, and cloud mode rejects the
local-only System-1 flags.
"""

from __future__ import annotations

import argparse
import contextlib
import shutil
from pathlib import Path
from typing import Any

import pytest

from rote import cli
from rote.compiler import Compiler, CompilerError
from rote.compiler.drivers import CompilerDriver, DriverResult
from rote.ir import load_pipeline

REPO = Path(__file__).resolve().parents[1]
EXAMPLES = REPO / "examples"


class _RubricSpy(CompilerDriver):
    """Records which rubric the agent got, then stops the compile."""

    name = "rubric-spy"

    def __init__(self) -> None:
        self.rubric: Path | None = None

    def is_available(self) -> tuple[bool, str]:
        return (True, "")

    async def run(
        self,
        skill_dir: Path,
        compiler_skill_dir: Path,
        work_dir: Path,
        extra_instructions: str | None = None,
        on_event: object = None,
    ) -> DriverResult:
        self.rubric = compiler_skill_dir
        raise _Stop


class _Stop(Exception):
    pass


def _skill(tmp_path: Path) -> Path:
    skill = tmp_path / "skill"
    shutil.copytree(EXAMPLES / "expense-audit" / "skill", skill)
    return skill


async def _rubric_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **kwargs: Any
) -> tuple[Path, list[str]]:
    spy = _RubricSpy()
    monkeypatch.setattr(Compiler, "select_driver", lambda self: spy)
    logs: list[str] = []
    kwargs.setdefault("system1_tiebreak", "none")
    compiler = Compiler(on_event=lambda e: logs.append(e.message), **kwargs)
    with pytest.raises(_Stop):
        await compiler.compile(_skill(tmp_path), tmp_path / "out")
    assert spy.rubric is not None
    return spy.rubric, logs


@pytest.mark.asyncio
async def test_a_flagless_compiler_runs_system1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rubric, _ = await _rubric_for(tmp_path, monkeypatch)
    assert rubric.name.startswith("rote-system1-rubric-")


@pytest.mark.asyncio
async def test_no_system1_runs_the_full_rubric(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rubric, _ = await _rubric_for(tmp_path, monkeypatch, system1=False)
    assert not rubric.name.startswith("rote-system1-rubric-")
    assert (rubric / "SKILL.md").read_text().startswith("---\nname: rote-compile\n")


@pytest.mark.asyncio
async def test_update_runs_the_full_rubric_and_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--update on the default path is the full-rubric agent, announced."""
    seen: dict[str, Any] = {}

    def plan_update(self: Compiler, output_dir: Path, text: str) -> Any:
        seen["planned"] = True
        raise CompilerError("stop after the mode decision")

    monkeypatch.setattr(Compiler, "_plan_update", plan_update)
    logs: list[str] = []
    compiler = Compiler(on_event=lambda e: logs.append(e.message))
    with pytest.raises(CompilerError, match="stop after the mode decision"):
        await compiler.compile(_skill(tmp_path), tmp_path / "out", update=True)
    assert seen == {"planned": True}
    assert any("--update runs the full-rubric agent" in m for m in logs)


@pytest.mark.asyncio
async def test_a_model_decider_never_runs_as_rules(tmp_path: Path) -> None:
    """--update cannot run System-1, so a model decider there is an error, not ignored."""
    skill = _skill(tmp_path)
    with pytest.raises(CompilerError, match="does not run System-1"):
        await Compiler(system1_decider="clef").compile(skill, tmp_path / "out", update=True)
    with pytest.raises(CompilerError, match="does not run System-1"):
        await Compiler(system1=False, system1_decider="clef").compile(skill, tmp_path / "out")


def _parsed(argv: list[str]) -> argparse.Namespace:
    out = ["--out", "o"] if argv[0] == "compile" else []
    return cli._build_parser().parse_args([*argv, *out])


@pytest.mark.parametrize("command", ["compile", "analyze"])
def test_the_only_flag_is_the_opt_out(command: str) -> None:
    assert _parsed([command, "s"]).system1 is None
    # No positive flag: System-1 is the default, so there is nothing to opt into.
    with pytest.raises(SystemExit):
        _parsed([command, "s", "--system1"])
    assert _parsed([command, "s", "--no-system1"]).system1 is False


@pytest.mark.parametrize("command", ["compile", "analyze"])
def test_the_cli_passes_the_flag_to_the_compiler(
    command: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[Any] = []
    real_init = Compiler.__init__

    def init(self: Compiler, *a: Any, **kw: Any) -> None:
        seen.append(kw.get("system1", "absent"))
        real_init(self, *a, **kw)
        raise _Stop

    monkeypatch.setattr(Compiler, "__init__", init)
    skill = _skill(tmp_path)
    for flags, expected in (([], None), (["--no-system1"], False)):
        argv = [command, str(skill), *flags]
        if command == "compile":
            argv += ["--local", "--runtime", "dbos", "--no-eval", "--out", str(tmp_path / "o")]
        with pytest.raises(_Stop):
            cli.main(argv)
        assert seen[-1] == expected


def test_a_decider_with_no_system1_is_a_usage_error(tmp_path: Path) -> None:
    for command in ("compile", "analyze"):
        argv = [command, str(_skill(tmp_path / command)), "--no-system1"]
        if command == "compile":
            argv += ["--out", str(tmp_path / "o")]
        assert cli.main([*argv, "--decider", "clef"]) == 2


def test_cloud_mode_rejects_the_local_system1_flags() -> None:
    assert cli._reject_local_only_cloud_flags(_parsed(["compile", "s"])) is None
    for flags, named in (
        (["--no-system1"], "--no-system1"),
        (["--decider", "jev"], "--decider"),
    ):
        message = cli._reject_local_only_cloud_flags(_parsed(["compile", "s", *flags]))
        assert message is not None and message.startswith(f"rote compile: {named} runs locally")


def test_system1_requires_a_signature_spec_on_every_judge(tmp_path: Path) -> None:
    """The legacy `signature` path emits only on Python runtimes; System-1's
    deliverable must emit everywhere, so a judge whose spec is missing or
    still the skeleton's (no prompt, no output fields) is an error."""
    from rote.compiler import system1
    from rote.compiler.verify import verify_deliverable

    work = tmp_path / "work"
    work.mkdir()
    system1.prepare(EXAMPLES / "deal-monitor" / "skill", work)
    judges = [n for n in load_pipeline(work / "pipeline.yaml").nodes if n.kind.value == "llm_judge"]
    assert len(judges) == 2

    def flagged(portable: bool) -> set[str]:
        v = verify_deliverable(work, run_tests=False, portable_judges=portable)
        return {f.node_id for f in v.findings if f.kind == "incomplete_signature_spec"}

    assert flagged(True) == {j.id for j in judges}
    assert flagged(False) == set()


def _fill_judges(work: Path) -> None:
    """What the text agent writes for each judge: a prompt and output fields."""
    import yaml

    from rote.compiler.system1.enforce import enforce

    doc = yaml.safe_load((work / "pipeline.yaml").read_text())
    text = {
        n["id"]: {
            "signature_spec": {
                "prompt": "Decide from {{ request }}.",
                "output_schema": {
                    "type": "object",
                    "properties": {"verdict": {"type": "string"}},
                    "required": ["verdict"],
                },
            }
        }
        for n in doc["nodes"]
        if n["kind"] == "llm_judge"
    }
    (work / "system1" / "text.yaml").write_text(yaml.safe_dump(text))
    enforce(work, final=True)


def test_the_skeleton_decides_each_judges_input_schema(tmp_path: Path) -> None:
    from rote.compiler import system1

    work = tmp_path / "work"
    work.mkdir()
    system1.prepare(EXAMPLES / "deal-monitor" / "skill", work)
    _fill_judges(work)
    pipeline = load_pipeline(work / "pipeline.yaml")
    for j in (n for n in pipeline.nodes if n.kind.value == "llm_judge"):
        assert j.signature is None  # one judge form: the spec
        spec = j.signature_spec
        assert spec is not None and spec.prompt == "Decide from {{ request }}."
        props = spec.input_schema["properties"]
        assert set(props) == set(j.inputs) == set(spec.input_schema["required"])
        assert props["request"] == pipeline.input.input_schema
        assert spec.output_schema["required"] == ["verdict"]


@pytest.mark.parametrize("runtime", ["cloudflare", "dbos", "temporal", "python"])
def test_a_system1_judge_pipeline_emits_on_every_runtime(runtime: str, tmp_path: Path) -> None:
    """System-1 plus a judge plus a TypeScript target: the emit that broke."""
    from rote.adapters import get_adapter
    from rote.compiler import system1
    from rote.compiler.verify import verify_deliverable

    work = tmp_path / "work"
    work.mkdir()
    system1.prepare(EXAMPLES / "deal-monitor" / "skill", work)
    _fill_judges(work)
    v = verify_deliverable(work, run_tests=False, portable_judges=True)
    assert not [f for f in v.findings if f.kind == "incomplete_signature_spec"]
    pipeline = load_pipeline(work / "pipeline.yaml")
    if runtime == "python" and pipeline.requires_durable_execution:
        pytest.skip("deal-monitor has a gate; the python adapter refuses gates by design")
    get_adapter(runtime, extracted_source_dir=work).emit(pipeline, tmp_path / "out")
    assert any((tmp_path / "out").rglob("*"))


@pytest.mark.asyncio
@pytest.mark.parametrize(("system1", "flagged"), [(None, True), (False, False)])
async def test_the_compiler_checks_judge_specs_on_the_system1_path_only(
    system1: bool | None, flagged: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[bool] = []
    import rote.compiler as compiler_mod

    real = compiler_mod.verify_deliverable

    def spy(*a: Any, **kw: Any) -> Any:
        seen.append(kw.get("portable_judges", False))
        return real(*a, **kw)

    monkeypatch.setattr(compiler_mod, "verify_deliverable", spy)

    class _Writes(_RubricSpy):
        async def run(
            self, skill_dir: Path, compiler_skill_dir: Path, work_dir: Path, **kw: Any
        ) -> DriverResult:  # noqa: E501
            if not (work_dir / "pipeline.yaml").is_file():
                shutil.copy(EXAMPLES / "expense-audit" / "expected" / "pipeline.yaml", work_dir)
            return DriverResult(work_dir / "pipeline.yaml", work_dir, self.name, {})

    monkeypatch.setattr(Compiler, "select_driver", lambda self: _Writes())
    with contextlib.suppress(CompilerError):
        await Compiler(system1=system1, run_generated_tests=False, system1_tiebreak="none").compile(
            _skill(tmp_path), tmp_path / "out"
        )
    assert seen and set(seen) == {flagged}


def _compiler_kwargs(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Every Compiler construction's kwargs; the construction then stops the command."""
    seen: list[dict[str, Any]] = []

    def init(self: Compiler, *a: Any, **kw: Any) -> None:
        seen.append(kw)
        raise _Stop

    monkeypatch.setattr(Compiler, "__init__", init)
    return seen


def _compile_argv(tmp_path: Path, *flags: str) -> list[str]:
    return [
        "compile",
        str(_skill(tmp_path)),
        "--local",
        "--runtime",
        "dbos",
        "--no-eval",
        "--out",
        str(tmp_path / "o"),
        *flags,
    ]


@pytest.mark.parametrize("command", ["compile", "analyze"])
def test_the_decider_is_a_layered_setting(
    command: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """flag > ROTE_DECIDER > project rote.yaml > user config > rules."""
    seen = _compiler_kwargs(monkeypatch)
    user = tmp_path / "user.yaml"
    project = tmp_path / "rote.yaml"
    monkeypatch.setenv("ROTE_CONFIG_PATH", str(user))
    monkeypatch.setenv("ROTE_PROJECT_CONFIG_PATH", str(project))
    skill = str(_skill(tmp_path))
    argv = _compile_argv(tmp_path / "c") if command == "compile" else ["analyze", skill]

    def decider(*flags: str) -> str:
        with pytest.raises(_Stop):
            cli.main([*argv, *flags])
        return str(seen[-1]["system1_decider"])

    assert decider() == "rules"
    user.write_text("decider: clef\n")
    assert decider() == "clef"
    project.write_text("decider: jev\n")
    assert decider() == "jev"
    monkeypatch.setenv("ROTE_DECIDER", "rules")
    assert decider() == "rules"
    assert decider("--decider", "clef") == "clef"


def test_a_bad_decider_setting_is_a_usage_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _compiler_kwargs(monkeypatch)
    monkeypatch.setenv("ROTE_DECIDER", "haiku")
    assert cli.main(_compile_argv(tmp_path)) == 2
    assert "ROTE_DECIDER: `decider: haiku` is not valid" in capsys.readouterr().err


def test_a_configured_decider_steps_aside_for_update_but_a_flag_does_not(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """--update runs the full-rubric agent. A standing `decider` setting must not
    break every update, and a `--decider` flag must not go silently unused."""
    seen = _compiler_kwargs(monkeypatch)
    monkeypatch.setenv("ROTE_DECIDER", "jev")
    with pytest.raises(_Stop):
        cli.main(_compile_argv(tmp_path / "a", "--update"))
    assert seen[-1]["system1_decider"] == "rules"
    err = capsys.readouterr().err
    assert "env ROTE_DECIDER sets `decider: jev`, which applies only to System-1" in err
    with pytest.raises(_Stop):
        cli.main(_compile_argv(tmp_path / "b", "--no-system1"))
    assert seen[-1]["system1_decider"] == "rules"
    assert len(seen) == 2
    for flags in (["--update"], ["--no-system1"]):
        assert cli.main(_compile_argv(tmp_path / flags[0][2:], *flags, "--decider", "jev")) == 2
    assert len(seen) == 2


def test_cloud_mode_ignores_a_configured_decider(monkeypatch: pytest.MonkeyPatch) -> None:
    """Like agent and model, a configured decider is a local preference: it does
    not trip the cloud mode's local-only flag check (which reads flags only)."""
    monkeypatch.setenv("ROTE_DECIDER", "clef")
    assert cli._reject_local_only_cloud_flags(_parsed(["compile", "s"])) is None


def test_rote_config_reports_the_decider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import json

    assert cli.main(["config", "--json"]) == 0
    setting = json.loads(capsys.readouterr().out)["settings"]["decider"]
    assert setting == {"value": "rules", "source": "built-in default"}
    monkeypatch.setenv("ROTE_DECIDER", "jev")
    assert cli.main(["config", "--json"]) == 0
    setting = json.loads(capsys.readouterr().out)["settings"]["decider"]
    assert setting == {"value": "jev", "source": "env ROTE_DECIDER"}


@pytest.mark.parametrize(
    ("written", "named"),
    [
        ({"prompt": "Decide from {{ request }}."}, "is missing output fields"),
        (
            {"output_schema": {"type": "object", "properties": {"v": {"type": "string"}}}},
            "is missing a prompt",
        ),
    ],
)
def test_each_half_of_a_judges_text_is_required(
    written: dict[str, Any], named: str, tmp_path: Path
) -> None:
    import yaml

    from rote.compiler import system1
    from rote.compiler.system1.enforce import enforce
    from rote.compiler.verify import verify_deliverable

    work = tmp_path / "work"
    work.mkdir()
    system1.prepare(EXAMPLES / "deal-monitor" / "skill", work)
    doc = yaml.safe_load((work / "pipeline.yaml").read_text())
    judges = [n["id"] for n in doc["nodes"] if n["kind"] == "llm_judge"]
    (work / "system1" / "text.yaml").write_text(
        yaml.safe_dump({j: {"signature_spec": written} for j in judges})
    )
    enforce(work, final=True)
    v = verify_deliverable(work, run_tests=False, portable_judges=True)
    messages = [f.message for f in v.findings if f.kind == "incomplete_signature_spec"]
    assert len(messages) == len(judges) and all(named in m for m in messages)


def test_the_step_after_a_gate_also_reads_what_the_gate_saw(tmp_path: Path) -> None:
    """A real expense-audit compile stubbed "file the report" because it only
    received the controller's decision, not the batch it was filing."""
    from rote.compiler import system1

    plan = system1.prepare(EXAMPLES / "expense-audit" / "skill", tmp_path)
    by_id = {n.id: n for n in plan.nodes}
    gate = next(n for n in plan.nodes if n.kind == "hitl_gate")
    gate_reads = {ref.split(".")[0] for ref in gate.inputs.values()} - {"pipeline"}
    after = [n for n in plan.nodes if f"{gate.id}.output" in n.inputs.values()]
    assert after and gate_reads
    for n in after:
        assert gate_reads <= {ref.split(".")[0] for ref in n.inputs.values()}
        assert "past the gate it also reads" in next(
            d.rule for d in n.decisions if d.field == "inputs"
        )
    assert all(by_id[g].kind != "hitl_gate" for g in gate_reads)


def test_judge_inputs_accept_what_their_upstream_really_returns(tmp_path: Path) -> None:
    """Emit deal-monitor's System-1 judges and validate real-shaped inputs.

    `score_new_opportunities_judge` reads the fanned classify judge's result,
    which binds as a LIST. Typing an upstream result as an object refused it.
    The fanned judge gets one element, whatever its shape."""
    import importlib.util
    import sys

    from rote.adapters import get_adapter
    from rote.compiler import system1

    work = tmp_path / "work"
    work.mkdir()
    system1.prepare(EXAMPLES / "deal-monitor" / "skill", work)
    _fill_judges(work)
    pipeline = load_pipeline(work / "pipeline.yaml")
    out = tmp_path / "out"
    get_adapter("dbos", extracted_source_dir=work).emit(pipeline, out)
    request = {"report_date": "2026-10-19"}
    cases = {
        # fanned over a thread list: one thread, of any shape
        "classify_warehouse_thread_pipeline_judge": {
            "search_gmail_outbound_quoting": {"id": "t1", "subject": "PO 4411"},
        },
        # the fanned judge's results: a list
        "score_new_opportunities_judge": {
            "classify_warehouse_thread_pipeline_judge": [{"verdict": "a"}, {"verdict": "b"}],
        },
    }
    sys.path.insert(0, str(out))
    try:
        for judge, inputs in cases.items():
            pascal = "".join(w.title() for w in judge.split("_"))
            path = out / "signatures" / f"{judge}.py"
            spec = importlib.util.spec_from_file_location(f"sig_{judge}", path)
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module  # pydantic resolves annotations through it
            spec.loader.exec_module(module)
            model = getattr(module, f"{pascal}Input")
            model.model_validate({"request": request, **inputs})
            # A string result validates too: nothing typed it yet.
            other = {k: "plain text" for k in inputs}
            model.model_validate({"request": request, **other})
    finally:
        sys.path.remove(str(out))


def test_a_list_a_judge_reads_whole_is_an_array(tmp_path: Path) -> None:
    from rote.compiler.system1 import decide
    from rote.compiler.system1.parse import parse_skill
    from rote.compiler.system1.skeleton import judge_input_schema

    unit = parse_skill(EXAMPLES / "deal-monitor" / "skill").units[0]
    node = decide.NodePlan("j", "llm_judge", "judge", unit)
    node.inputs = {"request": "pipeline.input", "rows": "fetch.output.items", "prev": "x.output"}
    props = judge_input_schema(node, {"type": "object"})["properties"]
    assert props["rows"]["type"] == "array"
    assert "type" not in props["prev"]
    node.fan_out, node.element_source = True, "fetch"
    node.inputs["others"] = "other.output.items"
    props = judge_input_schema(node, {"type": "object"})["properties"]
    assert "type" not in props["rows"]  # the fanned list: one element
    assert props["others"]["type"] == "array"  # another list, read whole
