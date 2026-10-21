"""Per-user state isolation stays complete as rote grows new state paths.

rote reads its per-user state paths from ``ROTE_*`` overrides (the app
registry, MCP registry and tokens, the cloud credential, layered config,
vendor specs and fixtures). ``rote.isolation.STATE_PATHS`` is the one list
of them that the test suite, the bench harness and ``scripts/isolated-rote``
all use. The set is derived from the source here, so a new override that
the list does not name fails.
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from rote.eval.empirical import default_corpus_path
from rote.eval.pricing import default_cache_dir
from rote.isolation import STATE_PATHS, isolated_env
from rote.serve.registry import default_registry_path

REPO = Path(__file__).resolve().parent.parent
WRAPPER = REPO / "scripts" / "isolated-rote"
BDR = REPO / "examples" / "bdr-outreach" / "expected" / "pipeline.yaml"

_STATE_VAR = re.compile(r"\bROTE_\w*(?:PATH|DIR|CONFIG)\b")


def _state_vars_read_by_rote() -> set[str]:
    return {
        name
        for path in (REPO / "src" / "rote").rglob("*.py")
        for name in _STATE_VAR.findall(path.read_text(encoding="utf-8"))
    }


def test_state_paths_names_every_state_override_the_source_reads() -> None:
    read = _state_vars_read_by_rote()
    assert {
        "ROTE_APPS_PATH",
        "ROTE_MCP_CONFIG",
        "ROTE_CLOUD_CRED_PATH",
        "ROTE_SERVE_REGISTRY_PATH",
        "ROTE_EVAL_CORPUS_PATH",
        "ROTE_PRICING_CACHE_DIR",
    } <= read
    assert read == set(STATE_PATHS)


#: Functions that use the home dir only to shorten paths for display.
_DISPLAY_ONLY_HOME = {("cli.py", "_cmd_config"), ("bench/runner.py", "_replacements")}


def _home_paths_without_an_override() -> list[str]:
    """Each ``Path.home()`` in src/rote whose function reads no STATE_PATHS
    override, directly or through a module constant naming one."""
    found = []
    src = REPO / "src" / "rote"
    for path in sorted(src.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        constants = {
            target.id
            for node in tree.body
            if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Constant)
            and node.value.value in STATE_PATHS
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        for func in ast.walk(tree):
            if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            calls_home = any(
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr == "home"
                and isinstance(n.func.value, ast.Name)
                and n.func.value.id == "Path"
                for n in ast.walk(func)
            )
            if not calls_home or (str(path.relative_to(src)), func.name) in _DISPLAY_ONLY_HOME:
                continue
            reads = {n.value for n in ast.walk(func) if isinstance(n, ast.Constant)} | {
                n.id for n in ast.walk(func) if isinstance(n, ast.Name)
            }
            if not (reads & set(STATE_PATHS) or reads & constants):
                found.append(f"{path.relative_to(src)}:{func.name}")
        module_level_home = [
            node
            for node in tree.body
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and "Path.home()" in (ast.get_source_segment(source, node) or "")
        ]
        found.extend(f"{path.relative_to(src)}:{n.lineno}" for n in module_level_home)
    return found


def test_every_home_dir_path_has_a_state_override() -> None:
    """The three gaps this list once had (the serve registry, the eval
    corpus, the price cache) were each a ``Path.home()`` with no override,
    which no amount of isolation could move."""
    assert _home_paths_without_an_override() == []


def test_conftest_redirects_every_state_path_into_the_test_tmp_dir(tmp_path: Path) -> None:
    # The autouse fixture has already run for this test.
    for name in STATE_PATHS:
        assert Path(os.environ[name]).is_relative_to(tmp_path), (name, os.environ[name])
    assert "ROTE_CLOUD_TOKEN" not in os.environ and "ROTE_CLOUD_URL" not in os.environ


def test_isolated_env_moves_every_path_and_drops_a_cloud_login(tmp_path: Path) -> None:
    base = {"PATH": "/usr/bin", "ROTE_CLOUD_TOKEN": "rote_live_key", "ROTE_CLOUD_URL": "https://x"}
    env = isolated_env(tmp_path / "state", base=base)
    assert env["PATH"] == "/usr/bin"
    assert "ROTE_CLOUD_TOKEN" not in env and "ROTE_CLOUD_URL" not in env
    for name in STATE_PATHS:
        assert Path(env[name]).is_relative_to(tmp_path / "state"), name


def _wrapper(*args: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    """The wrapper, on the interpreter running this suite: CI's newest-deps
    job installs rote without a repo .venv."""
    return subprocess.run(
        [str(WRAPPER), *args],
        env={**env, "ROTE_PYTHON": sys.executable},
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_an_isolated_emit_registers_the_app_in_the_isolated_registry(tmp_path: Path) -> None:
    """The real failure: ``rote emit`` appending to the developer's apps.json.

    conftest already points ROTE_APPS_PATH at a tmp file, which stands in
    for the real registry here: the wrapper must ignore it.
    """
    outer_apps = Path(os.environ["ROTE_APPS_PATH"])
    state = tmp_path / "state"
    out = tmp_path / "out"
    run = _wrapper(
        "emit", str(BDR), "--runtime", "dbos", "--out", str(out),
        env={**os.environ, "ROTE_ISOLATED_STATE": str(state)},
    )  # fmt: skip
    assert run.returncode == 0, run.stderr
    assert (out / "main.py").is_file()
    apps = json.loads((state / STATE_PATHS["ROTE_APPS_PATH"]).read_text())["apps"]
    assert [Path(a["path"]) for a in apps] == [out.resolve()]
    assert not outer_apps.exists()


def test_without_a_named_state_dir_the_temp_state_is_deleted(tmp_path: Path) -> None:
    scratch = tmp_path / "tmp"
    scratch.mkdir()
    env = {**os.environ, "TMPDIR": str(scratch)}
    env.pop("ROTE_ISOLATED_STATE", None)
    run = _wrapper("emit", str(BDR), "--runtime", "dbos", "--out", str(tmp_path / "out"), env=env)
    assert run.returncode == 0, run.stderr
    assert list(scratch.iterdir()) == []
    assert not Path(os.environ["ROTE_APPS_PATH"]).exists()


def test_the_wrapper_passes_rotes_exit_status(tmp_path: Path) -> None:
    args = ("emit", str(tmp_path / "missing.yaml"), "--out", str(tmp_path / "out"))
    plain = subprocess.run(
        [sys.executable, "-c", "import sys; from rote.cli import main; sys.exit(main())", *args],
        capture_output=True,
    )
    run = _wrapper(*args, env={**os.environ, "TMPDIR": str(tmp_path)})
    assert plain.returncode not in (0, None)
    assert run.returncode == plain.returncode


@pytest.mark.parametrize(
    ("env_var", "resolve", "home_default"),
    [
        ("ROTE_SERVE_REGISTRY_PATH", default_registry_path, ".rote/registry.json"),
        ("ROTE_EVAL_CORPUS_PATH", default_corpus_path, ".local/share/rote/eval-corpus.jsonl"),
        ("ROTE_PRICING_CACHE_DIR", default_cache_dir, ".cache/rote"),
    ],
)
def test_each_new_override_wins_and_the_default_is_unchanged(
    env_var: str,
    resolve: Callable[[], Path],
    home_default: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(env_var, str(tmp_path / "moved"))
    assert resolve() == tmp_path / "moved"
    monkeypatch.delenv(env_var)
    assert resolve() == Path.home() / home_default


def test_rote_register_without_a_flag_writes_the_overridden_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rote.cli import main
    from rote.serve.registry import Registry

    monkeypatch.setenv("DBOS_SYSTEM_DATABASE_URL", "postgresql://u:p@db:5432/rote")
    assert main(["register", str(BDR)]) == 0
    registry = Registry.load(Path(os.environ["ROTE_SERVE_REGISTRY_PATH"]))
    assert [Path(e.pipeline_yaml) for e in registry.entries] == [BDR.resolve()]
