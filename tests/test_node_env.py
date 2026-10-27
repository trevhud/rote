"""``Node.env``: the variables a node's code reads, declared in the IR (#267).

A compiled skill's driver runs with an allowlisted environment, so a
module's own variable reaches it only through this declaration. These cover
the IR boundary (which names may be declared), the static reader that finds
a module's reads, verification holding the declaration to the code, the run
plan carrying it, and the System-1 paths keeping it. The run itself is in
``tests/test_skill_commands.py`` and ``tests/test_skill_e2e.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from rote.compiler.env_reads import (
    ENV_RESERVED,
    ENV_UNDECLARED,
    declarable_reads,
    env_findings,
    env_reads,
)
from rote.env_vars import (
    BASE_VARS,
    CA_VARS,
    ENDPOINT_VARS,
    HELPER_KEYS,
    HELPER_KNOBS,
    PROXY_VARS,
    reserved_reason,
)
from rote.ir import Pipeline

REPO = Path(__file__).resolve().parent.parent


def _pipeline(env: Any = None, kind: str = "pure_function", **extra: Any) -> dict[str, Any]:
    node: dict[str, Any] = {
        "id": "write_report",
        "kind": kind,
        "description": "Write the report.",
        "impl": "extracted/report.py:write_report",
        "inputs": {"period": "pipeline.input.period"},
        **extra,
    }
    if env is not None:
        node["env"] = env
    return {
        "name": "reports",
        "version": "0.1.0",
        "input": {"type": "In", "required": ["period"]},
        "nodes": [node],
        "edges": [],
        "entry_nodes": ["write_report"],
        "exit_nodes": ["write_report"],
    }


# ───────── the IR boundary ─────────


def test_a_node_declares_the_variables_its_code_reads() -> None:
    pipeline = Pipeline.model_validate(_pipeline(["REPORT_DIR", "report_mode"]))
    assert pipeline.nodes[0].env == ["REPORT_DIR", "report_mode"]
    assert Pipeline.model_validate(_pipeline()).nodes[0].env is None


@pytest.mark.parametrize(
    "name",
    [
        "ROTE_MCP_CONFIG",
        "ROTE_MCP_TOKEN_DIR",
        "ROTE_SKILL_RUN_DIR",
        "ROTE_CLOUD_CRED_PATH",
        "ROTE_MCP_GH_URL",
        "ROTE_MCP_READ_ONLY",
        "ROTE_SKILL_COMMAND",
        "ROTE_MODEL_RATE",
        "ROTE_BASE_URL_RATE",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "OPENAI_API_KEY",
        "ANTHROPIC_BASE_URL",
        "OPENAI_BASE_URL",
        "HTTPS_PROXY",
        "no_proxy",
        "SSL_CERT_FILE",
        "REQUESTS_CA_BUNDLE",
        "PATH",
        "HOME",
        "LC_ALL",
        "PYTHONPATH",
        "PYTHONHOME",
        "LD_PRELOAD",
        "DYLD_INSERT_LIBRARIES",
        "XDG_CONFIG_HOME",
    ],
)
def test_a_name_the_allowlist_keeps_out_cannot_be_declared(name: str) -> None:
    with pytest.raises(ValueError, match=f"env cannot declare {name}"):
        Pipeline.model_validate(_pipeline([name]))


def test_every_name_the_skill_run_allowlist_handles_itself_is_reserved() -> None:
    """One list: every variable launch.child_env and trust.bindings pass, set or
    drop by name is one no declaration can reach (so a new one added there is
    covered here the day it lands)."""
    from rote.skill_target.trust import BINDINGS

    named = (
        set(BASE_VARS)
        | set(HELPER_KEYS)
        | set(ENDPOINT_VARS)
        | set(PROXY_VARS)
        | set(CA_VARS)
        | {var for var, _ in BINDINGS.values()}
        | {f"{prefix}X" for prefix in HELPER_KNOBS}
    )
    assert sorted(n for n in named if reserved_reason(n) is None) == []
    assert reserved_reason("EXPENSE_AUDIT_REPORT_DIR") is None


@pytest.mark.parametrize("name", ["1DIR", "REPORT-DIR", "A B", "X=1", "", "$HOME"])
def test_a_declared_name_must_be_a_variable_name(name: str) -> None:
    with pytest.raises(ValueError, match="must be a variable name"):
        Pipeline.model_validate(_pipeline([name]))


def test_env_is_only_for_nodes_that_run_code() -> None:
    judge = _pipeline(
        ["REPORT_DIR"],
        kind="llm_judge",
        impl=None,
        signature_spec={
            "input_schema": {"type": "object", "properties": {"period": {"type": "string"}}},
            "output_schema": {"type": "object", "properties": {"ok": {"type": "boolean"}}},
            "prompt": "Is {{ period }} fine?",
            "client": "anthropic",
        },
    )
    del judge["nodes"][0]["impl"]
    with pytest.raises(ValueError, match="env is only allowed on pure_function and external_call"):
        Pipeline.model_validate(judge)
    assert Pipeline.model_validate(_pipeline(["REPORT_DIR"], kind="external_call")).nodes[0].env


def test_a_name_is_declared_once() -> None:
    with pytest.raises(ValueError, match="env names REPORT_DIR more than once"):
        Pipeline.model_validate(_pipeline(["REPORT_DIR", "REPORT_DIR"]))


# ───────── reading the code ─────────


def test_every_literal_read_form_is_found() -> None:
    source = (
        "import os\n"
        "import os as o\n"
        "from os import environ as E, getenv\n"
        "DIR_ENV = 'BY_CONSTANT'\n"
        "def f():\n"
        "    a = os.environ['BY_SUBSCRIPT']\n"
        "    b = o.environ.get('BY_GET', 'x')\n"
        "    c = E.pop('BY_POP', None)\n"
        "    d = os.getenv('BY_GETENV')\n"
        "    e = getenv('BY_IMPORTED_GETENV')\n"
        "    g = 'BY_IN' in os.environ\n"
        "    h = os.environ.get(DIR_ENV)\n"
        "    os.environ['WRITTEN'] = 'v'\n"
        "    return a, b, c, d, e, g, h\n"
    )
    found = env_reads(source)
    assert sorted(found.names) == [
        "BY_CONSTANT",
        "BY_GET",
        "BY_GETENV",
        "BY_IMPORTED_GETENV",
        "BY_IN",
        "BY_POP",
        "BY_SUBSCRIPT",
    ]
    assert found.dynamic == []


def test_a_read_no_name_describes_is_dynamic() -> None:
    source = (
        "import os\n"
        "NAME = 'A'\n"
        "NAME = 'B'\n"  # bound twice: not a constant
        "def f(key):\n"
        "    x = os.environ[key]\n"
        "    y = os.environ.get(NAME)\n"
        "    z = dict(os.environ)\n"
        "    w = os.environ.copy()\n"
        "    return x, y, z, w\n"
    )
    found = env_reads(source)
    assert found.names == {}
    assert found.dynamic == [5, 6, 7, 8]


def test_a_module_that_never_imports_os_reads_nothing() -> None:
    # `environ` here is the caller's own dict, not the process environment.
    assert env_reads("def f(environ):\n    return environ['X']\n").names == {}


# ───────── verification ─────────


def _deliverable(tmp_path: Path, env: Any, modules: dict[str, str]) -> Pipeline:
    for rel, text in modules.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    data = _pipeline(env)
    (tmp_path / "pipeline.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    return Pipeline.model_validate(data)


REPORT = (
    "import os\n"
    "from extracted.paths import base\n\n"
    "REPORT_DIR_ENV = 'REPORTS_REPORT_DIR'\n\n"
    "def write_report(period):\n"
    "    home = os.environ['HOME']\n"  # always passed: no declaration needed
    "    return {'dir': os.environ.get(REPORT_DIR_ENV) or base(), 'home': home}\n"
)
PATHS = "import os\n\ndef base():\n    return os.getenv('REPORTS_BASE', 'reports')\n"


def test_a_read_the_node_does_not_declare_goes_back_for_repair(tmp_path: Path) -> None:
    pipeline = _deliverable(
        tmp_path,
        ["REPORTS_REPORT_DIR"],
        {"extracted/report.py": REPORT, "extracted/paths.py": PATHS},
    )
    findings, questions = env_findings(pipeline, tmp_path)
    # The imported helper module's read counts: it runs in the same process.
    assert [(f.kind, f.severity, f.node_id, f.path) for f in findings] == [
        (ENV_UNDECLARED, "error", "write_report", "extracted/paths.py")
    ]
    assert "extracted/paths.py:4 reads $REPORTS_BASE" in findings[0].message
    assert "`env: [REPORTS_BASE, REPORTS_REPORT_DIR]`" in findings[0].message
    assert questions == []
    assert declarable_reads(pipeline.nodes[0], tmp_path) == ["REPORTS_BASE", "REPORTS_REPORT_DIR"]
    # Negative half: declaring both closes it.
    full = _deliverable(tmp_path, ["REPORTS_BASE", "REPORTS_REPORT_DIR"], {})
    assert env_findings(full, tmp_path) == ([], [])


def test_verification_reports_an_undeclared_read(tmp_path: Path) -> None:
    from rote.compiler.verify import verify_deliverable

    _deliverable(tmp_path, None, {"extracted/report.py": REPORT, "extracted/paths.py": PATHS})
    verification = verify_deliverable(tmp_path, run_tests=False)
    kinds = sorted(
        (f.kind, f.message.split(" reads ")[1].split(" ")[0])
        for f in verification.errors
        if f.kind == ENV_UNDECLARED
    )
    assert kinds == [(ENV_UNDECLARED, "$REPORTS_BASE"), (ENV_UNDECLARED, "$REPORTS_REPORT_DIR")]
    assert not verification.ok
    repair = verification.repair_instruction(tmp_path)
    assert repair is not None and "Add `env: [REPORTS_BASE]` to write_report" in repair


def test_a_reserved_read_cannot_be_declared_away(tmp_path: Path) -> None:
    module = "import os\n\ndef write_report(period):\n    return os.environ['ROTE_MCP_CONFIG']\n"
    pipeline = _deliverable(tmp_path, None, {"extracted/report.py": module})
    findings, _ = env_findings(pipeline, tmp_path)
    assert [f.kind for f in findings] == [ENV_RESERVED]
    assert "Take the value as an input or a constant instead" in findings[0].message
    assert declarable_reads(pipeline.nodes[0], tmp_path) == []


def test_a_dynamic_read_is_listed_for_review_not_failed(tmp_path: Path) -> None:
    module = "import os\n\ndef write_report(period):\n    return os.environ[period]\n"
    pipeline = _deliverable(tmp_path, None, {"extracted/report.py": module})
    findings, questions = env_findings(pipeline, tmp_path)
    assert findings == []
    assert [q.node_id for q in questions] == ["write_report"]
    assert "extracted/report.py:4 uses the environment by a name" in questions[0].text


def test_the_expense_audit_example_declares_what_its_report_reads() -> None:
    from rote.ir import load_pipeline

    compiled = REPO / "examples" / "expense-audit" / "expected"
    pipeline = load_pipeline(compiled / "pipeline.yaml")
    assert env_findings(pipeline, compiled) == ([], [])
    assert pipeline.node_by_id("file_report").env == ["EXPENSE_AUDIT_REPORT_DIR"]


# ───────── the plan ─────────


def test_the_plan_carries_the_declaration_only_when_there_is_one() -> None:
    from rote.plan import build_plan

    (step,) = build_plan(Pipeline.model_validate(_pipeline(["REPORT_DIR"])), mcp="live")["steps"]
    assert step["kind"] == "call" and step["env"] == ["REPORT_DIR"]
    (plain,) = build_plan(Pipeline.model_validate(_pipeline()), mcp="live")["steps"]
    assert "env" not in plain


def test_the_driver_environment_takes_the_declared_names_from_the_caller() -> None:
    from rote.skill_target.launch import child_env

    bound = {
        "mcp_config": "/m.json",
        "token_dir": "/t",
        "run_dir": "/r",
        "cloud_cred": "/c.json",
        "proxy": {},
        "ca": {},
        "env": ["REPORT_DIR"],
    }
    environ = {"PATH": "/bin", "REPORT_DIR": "/finance", "OTHER_DIR": "/elsewhere"}
    env, _ = child_env(environ, {"steps": []}, bound)
    assert env["REPORT_DIR"] == "/finance"
    assert "OTHER_DIR" not in env
    # Declared but unset in the caller: absent, as on every other runtime.
    env, _ = child_env({"PATH": "/bin"}, {"steps": []}, bound)
    assert "REPORT_DIR" not in env


# ───────── System-1 ─────────


def test_parallel_generation_declares_what_a_written_module_reads(tmp_path: Path) -> None:
    """On the parallel path a model writes the module and code writes
    pipeline.yaml, so code reads the module and declares its variables:
    nothing a re-ask could add."""
    from rote.compiler import system1
    from rote.compiler.grounding import Grounding
    from rote.compiler.system1 import generate
    from rote.compiler.system1.generate import Context

    work = tmp_path / "work"
    work.mkdir()
    plan = system1.prepare(REPO / "examples" / "expense-audit" / "skill", work)
    ctx = Context.load(work, plan, Grounding())
    node = next(n for n in ctx.pipeline.nodes if n.kind.value == "pure_function")
    assert node.impl is not None
    module = work / node.impl.partition(":")[0]
    module.parent.mkdir(parents=True, exist_ok=True)
    module.write_text(
        f"import os\n\ndef {node.id}(**kw):\n    return os.environ.get('AUDIT_DIR')\n",
        encoding="utf-8",
    )
    assert generate.text_entries(ctx, {})[node.id]["env"] == ["AUDIT_DIR"]
    module.write_text(f"def {node.id}(**kw):\n    return None\n", encoding="utf-8")
    assert "env" not in generate.text_entries(ctx, {}).get(node.id, {})


def test_an_env_the_agent_writes_survives_enforcement(tmp_path: Path) -> None:
    """env is the agent's to write (it writes the code), so restoring System-1's
    decisions must keep it, or the repair verification asks for could never land."""
    from rote.compiler import system1
    from rote.compiler.system1.enforce import enforce

    work = tmp_path / "work"
    work.mkdir()
    system1.prepare(REPO / "examples" / "expense-audit" / "skill", work)
    doc = yaml.safe_load((work / "pipeline.yaml").read_text())
    target = next(n for n in doc["nodes"] if n["kind"] == "pure_function")
    target["env"] = ["AUDIT_DIR"]
    (work / "pipeline.yaml").write_text(yaml.safe_dump(doc, sort_keys=False))
    enforce(work)
    after = yaml.safe_load((work / "pipeline.yaml").read_text())
    assert next(n for n in after["nodes"] if n["id"] == target["id"])["env"] == ["AUDIT_DIR"]
