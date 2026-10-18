"""Check a compiler deliverable before the compile calls itself done.

The agent's deliverable is a directory: ``pipeline.yaml`` plus the
``extracted/`` modules and ``tests/`` it wrote. Each part makes promises
that are cheap to check and expensive to discover later:

* the IR validates (:func:`rote.ir.load_pipeline`);
* every module the IR points at parses on the oldest supported Python,
  defines the named symbol, and can take the node's payload
  (:func:`rote.contracts.check_contracts`);
* every ``pure_function`` is either implemented or a stub that says why
  it is one, and every implemented one is exercised by a generated test;
* the generated tests pass.

The orchestrator runs this on every natural stop of the agent and feeds
the rendered problems back into the SAME conversation
(:meth:`Verification.repair_instruction`), so the model that wrote the
code fixes it while it still has the skill in context. It runs once more
after the agent is done, and the CLI reports that final verdict.

Running the generated tests executes agent-written code. That is the
point of them, and it is no new exposure: the emitted pipeline runs the
same modules. They run in a subprocess, never in rote's own process.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from rote.contracts import (
    ContractFinding,
    NodeImplementation,
    check_contracts,
    implementation_status,
)
from rote.ir import NodeKind, Pipeline, load_pipeline

#: Wall-clock cap on one generated-test run. Pure-function tests are
#: millisecond-scale, so hitting this means an infinite loop, which the
#: agent can fix like any other failure.
GENERATED_TESTS_TIMEOUT_SECONDS = 120.0

#: How much pytest output a repair instruction carries. The tail holds the
#: failure summary; the whole log would re-bill every line on each turn.
_TEST_OUTPUT_CHARS = 6000

TestStatus = Literal["passed", "failed", "unavailable"]


@dataclass(frozen=True)
class GeneratedTests:
    """One run of the agent-written ``tests/``."""

    status: TestStatus
    summary: str
    output: str = ""

    @property
    def passed(self) -> bool:
        return self.status == "passed"

    def as_dict(self) -> dict[str, object]:
        return {"passed": self.passed, "status": self.status, "summary": self.summary}


def _pytest_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("pytest") is not None


def run_generated_tests(artifact_dir: Path | str) -> GeneratedTests | None:
    """Run ``<artifact_dir>/tests`` with pytest, if there is anything to run.

    ``python -m pytest`` from the artifact dir puts that dir on
    ``sys.path``, which is what lets a test say ``from extracted.parsers
    import parse_x`` with no conftest. ``unavailable`` (pytest is not
    installed alongside rote) is reported, never treated as a pass, and
    never sent to the agent: it cannot fix the environment.
    """
    root = Path(artifact_dir)
    tests_dir = root / "tests"
    if not tests_dir.is_dir() or not any(tests_dir.glob("test_*.py")):
        return None
    if not _pytest_available():
        return GeneratedTests(
            status="unavailable",
            summary=f"pytest is not installed for {sys.executable}; generated tests not run",
        )
    try:
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests",
                "-q",
                "--no-header",
                "-p",
                "no:cacheprovider",
                "--tb=short",
            ],
            cwd=root,
            # No bytecode cache. A repair round can rewrite a module within
            # the same second at the same size (`+ 2` -> `* 2`), and then
            # the stale .pyc from the previous run is what gets imported:
            # the fixed code tests red. It also keeps __pycache__ out of
            # the compiled output.
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            capture_output=True,
            text=True,
            timeout=GENERATED_TESTS_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return GeneratedTests(
            status="failed",
            summary=f"timed out after {GENERATED_TESTS_TIMEOUT_SECONDS:.0f}s",
            output="The generated tests did not finish; look for an unbounded loop.",
        )
    text = (proc.stdout or "") + (proc.stderr or "")
    lines = text.strip().splitlines()
    return GeneratedTests(
        status="passed" if proc.returncode == 0 else "failed",
        summary=lines[-1] if lines else f"pytest exited {proc.returncode} with no output",
        output=text[-_TEST_OUTPUT_CHARS:],
    )


def _tested_symbols(artifact_dir: Path) -> set[str]:
    """Every identifier the generated tests mention.

    A deliberately loose reading: a test that names the function at all
    counts. The point is to catch an implementation nobody wrote a test
    for, not to grade the tests.
    """
    names: set[str] = set()
    tests_dir = artifact_dir / "tests"
    if not tests_dir.is_dir():
        return names
    for path in tests_dir.rglob("*.py"):
        try:
            names.update(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", path.read_text(encoding="utf-8")))
        except OSError:
            continue
    return names


def _deliverable_findings(
    pipeline: Pipeline, artifact_dir: Path, implementations: list[NodeImplementation]
) -> list[ContractFinding]:
    """The rules that hold for a fresh compile, but not for any IR on disk.

    ``check_contracts`` also runs on hand-written and older pipelines at
    ``rote emit`` time, where an unwritten module is the documented stub
    path. A compile is held to more: a ``pure_function`` is code the
    skill describes, so the agent either writes it or says why not.
    """
    findings: list[ContractFinding] = []
    tested = _tested_symbols(artifact_dir)
    kinds = {n.id: n.kind for n in pipeline.nodes}
    for impl in implementations:
        if kinds.get(impl.node_id) is not NodeKind.PURE_FUNCTION:
            continue
        rel, _, symbol = impl.impl.partition(":")
        if impl.status == "absent":
            findings.append(
                ContractFinding(
                    node_id=impl.node_id,
                    kind="pure_function_not_written",
                    severity="error",
                    message=(
                        f"{impl.impl} does not exist. Write the function, or a stub "
                        f"whose NotImplementedError message says what the skill "
                        f"leaves unstated (implementation.md)"
                    ),
                    path=rel,
                )
            )
        elif impl.status == "stub" and not impl.reason:
            findings.append(
                ContractFinding(
                    node_id=impl.node_id,
                    kind="unexplained_stub",
                    severity="error",
                    message=(
                        f"{symbol}() is a stub with no reason. A pure_function stays "
                        f"a stub only when the skill leaves a fact unstated; raise "
                        f"NotImplementedError('<what is missing and why>') so the "
                        f"decision is reviewable (implementation.md)"
                    ),
                    path=rel,
                )
            )
        elif impl.status == "implemented" and symbol not in tested:
            findings.append(
                ContractFinding(
                    node_id=impl.node_id,
                    kind="untested_implementation",
                    severity="error",
                    message=(
                        f"{symbol}() is implemented but no file under tests/ calls "
                        f"it. Add cases derived from the skill's own rules and "
                        f"examples (implementation.md)"
                    ),
                    path=rel,
                )
            )
    return findings


@dataclass(frozen=True)
class Verification:
    """The verdict on one deliverable directory."""

    pipeline: Pipeline | None
    pipeline_error: str | None = None
    findings: list[ContractFinding] = field(default_factory=list)
    implementations: list[NodeImplementation] = field(default_factory=list)
    tests: GeneratedTests | None = None

    @property
    def errors(self) -> list[ContractFinding]:
        return [f for f in self.findings if f.severity == "error"]

    @property
    def ok(self) -> bool:
        return (
            self.pipeline_error is None
            and not self.errors
            and (self.tests is None or self.tests.status != "failed")
        )

    def repair_instruction(self, artifact_dir: Path) -> str | None:
        """The user turn that asks the agent to fix what failed, or ``None``.

        Verbatim problems, the paths they are in, and a scope rule: fix
        only what is named. The scope rule is also the cost rule, since
        every turn of a resumed conversation re-reads the whole context.
        """
        if self.pipeline_error is not None:
            return (
                f"The pipeline.yaml you wrote fails IR validation. Fix "
                f"{artifact_dir / 'pipeline.yaml'} so it validates; change only what "
                f"the errors name, then end your turn.\n\n"
                f"Validation errors:\n{self.pipeline_error}"
            )
        if self.ok:
            return None
        parts = [
            f"rote checked the deliverable in {artifact_dir} and found problems. "
            f"Fix each one; change only what they name, then end your turn. Do not "
            f"weaken an assertion that encodes a rule from the skill, and do not turn "
            f"a working implementation into a stub to silence a failure: if a test "
            f"and the implementation disagree, the skill decides which is wrong. "
            f"Leave compile-report.md alone unless a decision it records changed."
        ]
        if self.errors:
            parts.append(
                "Contract problems:\n"
                + "\n".join(f"- {f.node_id}: {f.message}" for f in self.errors)
            )
        if self.tests is not None and self.tests.status == "failed":
            parts.append(
                f"The generated tests fail ({self.tests.summary}). pytest output, "
                f"run as `python -m pytest tests` from {artifact_dir}:\n\n"
                f"{self.tests.output}"
            )
        return "\n\n".join(parts)


#: The subdirectories a deliverable's code lives in, and so the ones an
#: update run's partial work dir is overlaid on.
_CODE_DIRS = ("extracted", "signatures", "tests")


def verify_deliverable(
    artifact_dir: Path | str, *, base_dir: Path | str | None = None, run_tests: bool = True
) -> Verification:
    """Verify the deliverable in ``artifact_dir`` (the dir holding pipeline.yaml).

    ``base_dir`` is for ``compile --update``: the agent's work dir then
    holds only what it rewrote, and the modules it left alone live in the
    previous output. Verifying the work dir alone would report every
    untouched module as missing, so the check runs on the previous code
    with the work dir laid over it, which is exactly what the merge into
    the output will produce.
    """
    if base_dir is None:
        return _verify(Path(artifact_dir), run_tests=run_tests)
    import shutil
    import tempfile

    with tempfile.TemporaryDirectory(prefix="rote-verify-") as tmp:
        overlay = Path(tmp)
        for source in (Path(base_dir), Path(artifact_dir)):
            for sub in _CODE_DIRS:
                if (source / sub).is_dir():
                    shutil.copytree(source / sub, overlay / sub, dirs_exist_ok=True)
        produced = Path(artifact_dir) / "pipeline.yaml"
        if produced.is_file():
            shutil.copy2(produced, overlay / "pipeline.yaml")
        return _verify(overlay, run_tests=run_tests)


def _verify(root: Path, *, run_tests: bool) -> Verification:
    try:
        pipeline = load_pipeline(root / "pipeline.yaml")
    except Exception as e:  # noqa: BLE001 — any load failure is the agent's to fix
        return Verification(pipeline=None, pipeline_error=str(e))
    implementations = implementation_status(pipeline, root)
    findings = check_contracts(pipeline, root) + _deliverable_findings(
        pipeline, root, implementations
    )
    tests = run_generated_tests(root) if run_tests else None
    return Verification(
        pipeline=pipeline,
        findings=findings,
        implementations=implementations,
        tests=tests,
    )
