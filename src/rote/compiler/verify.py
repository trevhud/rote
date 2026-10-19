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

Running the generated tests executes agent-written code, automatically,
on the machine running ``rote compile``, before anyone has reviewed it.
That is new exposure: the agent wrote that code from a SKILL.md that may
be third-party and is untrusted input (invariant #7), and a skill can
steer the agent into a test that reads a credential and sends it
somewhere. So the tests run in a subprocess with an environment built
from an allowlist (:func:`generated_test_env`): no inherited variables
beyond PATH and locale, so no ``ANTHROPIC_API_KEY``,
``CLOUDFLARE_API_TOKEN``, or ``ROTE_CLOUD_TOKEN``, and a throwaway HOME,
so ``~/.local/share/rote/cloud.json`` and other dotfiles resolve to an
empty directory. That is a credential boundary, not a sandbox: the
process can still open absolute paths and reach the network. For a skill
you do not trust, ``rote compile --no-verify-tests`` skips running them
(the static checks still run), and the tests can be reviewed and run by
hand.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
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


#: Inherited variables a generated test may see. An allowlist, not a
#: scrub list: ``build_subscription_env`` removes the two Anthropic keys
#: because it must keep everything else a ``claude`` process needs, and
#: the suite's own guard removes the vendor keys it knows about. Neither
#: list knows every secret a developer exports (CLOUDFLARE_API_TOKEN,
#: ROTE_CLOUD_TOKEN, AWS_*, GITHUB_TOKEN...), and agent-written tests need
#: none of them. Pure-function tests need a PATH and a locale.
_INHERITED_TEST_ENV = ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "TZ", "SYSTEMROOT")


def generated_test_env(artifact_dir: Path, home: Path) -> dict[str, str]:
    """The whole environment the generated tests run with.

    ``home`` is an empty throwaway directory, so ``~/.local/share/rote``,
    ``~/.aws`` and every other dotfile resolve to nothing. The rest:

    * ``PYTHONPATH`` is the artifact dir, so ``from extracted.x import f``
      resolves without a conftest;
    * ``PYTHONDONTWRITEBYTECODE``: a repair round can rewrite a module
      within the same second at the same size (``+ 2`` to ``* 2``), and a
      stale .pyc then tests the old code. It also keeps __pycache__ out of
      the compiled output;
    * ``PYTHONNOUSERSITE`` and ``PYTEST_DISABLE_PLUGIN_AUTOLOAD``: the
      tests are plain pytest over the stdlib, so nothing from the user's
      site-packages or pytest plugins needs to load into them.
    """
    env = {k: os.environ[k] for k in _INHERITED_TEST_ENV if k in os.environ}
    env.update(
        HOME=str(home),
        USERPROFILE=str(home),
        TMPDIR=str(home),
        XDG_CONFIG_HOME=str(home / ".config"),
        XDG_DATA_HOME=str(home / ".local" / "share"),
        PYTHONPATH=str(artifact_dir),
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONNOUSERSITE="1",
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
    )
    return env


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
        with tempfile.TemporaryDirectory(prefix="rote-test-home-") as home:
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
                env=generated_test_env(root, Path(home)),
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
            text = path.read_text(encoding="utf-8", errors="replace")
            names.update(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text))
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
        if impl.status == "absent" and (artifact_dir / rel).is_file():
            # The module exists but does not parse, decode, or define the
            # symbol; check_contracts already says which.
            continue
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


REPORT_FILENAME = "compile-report.md"

#: ``# skill: "<verbatim text>"``: the skill sentence that determines the
#: value or branch next to it (implementation.md).
_CITATION_RE = re.compile(r'#\s*skill:\s*"(?P<quote>[^"]+)"')
#: ``# rote: assumption: <what was chosen>``: an outcome the skill does
#: not state, decided by the agent and owed to the report's Open questions.
_ASSUMPTION_RE = re.compile(r"#\s*rote:\s*assumption:\s*(?P<text>.+)")


def _normalized(text: str) -> str:
    """Lowercase, markdown emphasis and code marks dropped, whitespace collapsed."""
    return " ".join(re.sub(r"[*_`]", "", text).lower().split())


def _skill_text(skill_dir: Path) -> str:
    parts = [
        p.read_text(encoding="utf-8", errors="replace") for p in sorted(skill_dir.rglob("*.md"))
    ]
    return _normalized("\n".join(parts))


#: A bold line standing in for a heading (``**Open questions:**``).
_BOLD_HEADING_LEVEL = 99


def _open_questions(report: str) -> list[str]:
    """The entries under the report's Open questions heading.

    An entry is a list item or a sub-heading (``### OQ-1: ...``). A real
    report used one sub-heading per question, which an items-only reading
    took for an empty section: the agent was sent back twice to fix a
    report that was fine, and the compile failed. The section ends at the
    next heading of the same or a higher level.
    """
    items: list[str] = []
    level: int | None = None  # the Open questions heading's level, while inside
    for line in report.splitlines():
        stripped = line.strip()
        hashes = re.match(r"^(#+)\s+(.*)", stripped)
        bold = stripped.startswith("**") and stripped.rstrip(":").endswith("**")
        if hashes or bold:
            this = len(hashes.group(1)) if hashes else _BOLD_HEADING_LEVEL
            if level is not None and this > level:
                items.append(stripped)
                continue
            level = this if "open question" in stripped.lower() else None
            continue
        if level is not None and re.match(r"^([-*+]|\d+[.)])\s+\S", stripped):
            items.append(stripped)
    return items


def _provenance_findings(root: Path, skill_dir: Path | None) -> list[ContractFinding]:
    """Every outcome traces to the skill, or is declared as an assumption.

    A real compile decided that an expense over its category cap is a
    ``block``. The skill lists the rule but never says block or review,
    the hand-written reference says review, and the agent's own tests
    encoded its choice, so they passed. Two checks make such a choice
    visible instead:

    * a ``# skill: "..."`` citation must quote text the skill contains,
      so a citation cannot be invented to cover a guess;
    * a ``# rote: assumption:`` comment means the report's Open questions
      must not be empty, so the guess reaches a reviewer.
    """
    findings: list[ContractFinding] = []
    skill = _skill_text(skill_dir) if skill_dir is not None and skill_dir.is_dir() else None
    assumptions: list[tuple[str, str]] = []
    for path in sorted((root / "extracted").rglob("*.py")) if (root / "extracted").is_dir() else []:
        rel = str(path.relative_to(root))
        text = path.read_text(encoding="utf-8", errors="replace")
        assumptions += [(rel, m.group("text").strip()) for m in _ASSUMPTION_RE.finditer(text)]
        if skill is None:
            continue
        for m in _CITATION_RE.finditer(text):
            # "..." (or an ellipsis) elides text between verbatim fragments.
            # A long quote wraps across comment lines; the "#" that starts
            # each continuation line is not part of the skill's text.
            raw = re.sub(r"\n\s*#\s?", " ", m.group("quote")).replace("…", "...")
            fragments = [_normalized(f) for f in raw.split("...")]
            if not all(f in skill for f in fragments if f):
                findings.append(
                    ContractFinding(
                        node_id="(provenance)",
                        kind="citation_not_in_skill",
                        severity="error",
                        message=(
                            f'{rel} cites the skill as "{m.group("quote")}", but the skill '
                            f"contains no such text. Quote it verbatim, or if the skill does "
                            f"not state this outcome, mark it '# rote: assumption: ...' and "
                            f"list it under Open questions"
                        ),
                        path=rel,
                    )
                )
    if assumptions:
        report = root / REPORT_FILENAME
        listed = (
            _open_questions(report.read_text(encoding="utf-8", errors="replace"))
            if report.is_file()
            else []
        )
        if not listed:
            sample = "; ".join(f"{rel}: {text}" for rel, text in assumptions[:3])
            findings.append(
                ContractFinding(
                    node_id="(provenance)",
                    kind="unlisted_assumption",
                    severity="error",
                    message=(
                        f"the extracted code records {len(assumptions)} assumption(s) the "
                        f"skill does not state ({sample}), but {REPORT_FILENAME} lists no "
                        f"Open questions. Add one entry per assumption"
                    ),
                    path=REPORT_FILENAME,
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
            f"Leave compile-report.md alone unless a problem below names it or a "
            f"decision it records changed."
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
    artifact_dir: Path | str,
    *,
    base_dir: Path | str | None = None,
    run_tests: bool = True,
    skill_dir: Path | str | None = None,
) -> Verification:
    """Verify the deliverable in ``artifact_dir`` (the dir holding pipeline.yaml).

    ``base_dir`` is for ``compile --update``: the agent's work dir then
    holds only what it rewrote, and the modules it left alone live in the
    previous output. Verifying the work dir alone would report every
    untouched module as missing, so the check runs on the previous code
    with the work dir laid over it, which is exactly what the merge into
    the output will produce.

    ``skill_dir`` is the source skill. With it, every ``# skill: "..."``
    citation in the extracted modules must quote text the skill actually
    contains (see :func:`_provenance_findings`).
    """
    try:
        return _verify_or_raise(artifact_dir, base_dir, run_tests, skill_dir)
    except Exception as e:  # noqa: BLE001 (a crash here must not cost the paid run)
        # Verification runs inside the driver's repair loop, where only
        # DriverError is caught, and before the work dir leaves its temp
        # home. An exception from either place would delete the agent's
        # work. Whatever went wrong becomes one more problem to report and,
        # if it is the agent's doing, to repair.
        return Verification(
            pipeline=None,
            findings=[
                ContractFinding(
                    node_id="(deliverable)",
                    kind="verification_error",
                    severity="error",
                    message=f"rote could not check the deliverable: {type(e).__name__}: {e}",
                )
            ],
        )


def _verify_or_raise(
    artifact_dir: Path | str,
    base_dir: Path | str | None,
    run_tests: bool,
    skill_dir: Path | str | None,
) -> Verification:
    skill = Path(skill_dir) if skill_dir is not None else None
    if base_dir is None:
        return _verify(Path(artifact_dir), run_tests=run_tests, skill_dir=skill)
    import shutil
    import tempfile

    with tempfile.TemporaryDirectory(prefix="rote-verify-") as tmp:
        overlay = Path(tmp)
        for source in (Path(base_dir), Path(artifact_dir)):
            for sub in _CODE_DIRS:
                if (source / sub).is_dir():
                    shutil.copytree(source / sub, overlay / sub, dirs_exist_ok=True)
        for source in (Path(base_dir), Path(artifact_dir)):
            if (source / REPORT_FILENAME).is_file():
                shutil.copy2(source / REPORT_FILENAME, overlay / REPORT_FILENAME)
        produced = Path(artifact_dir) / "pipeline.yaml"
        if produced.is_file():
            shutil.copy2(produced, overlay / "pipeline.yaml")
        return _verify(overlay, run_tests=run_tests, skill_dir=skill)


def _verify(root: Path, *, run_tests: bool, skill_dir: Path | None) -> Verification:
    try:
        pipeline = load_pipeline(root / "pipeline.yaml")
    except Exception as e:  # noqa: BLE001 (any load failure is the agent's to fix)
        return Verification(pipeline=None, pipeline_error=str(e))
    implementations = implementation_status(pipeline, root)
    findings = (
        check_contracts(pipeline, root)
        + _deliverable_findings(pipeline, root, implementations)
        + _provenance_findings(root, skill_dir)
    )
    tests = run_generated_tests(root) if run_tests else None
    return Verification(
        pipeline=pipeline,
        findings=findings,
        implementations=implementations,
        tests=tests,
    )
