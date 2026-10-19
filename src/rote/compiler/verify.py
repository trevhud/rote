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
* the generated tests pass;
* the target runtime's adapter emits it (:func:`emit_findings`, a dry
  emit into a throwaway dir), so a compile never declares success on an
  IR that ``rote emit`` then refuses.

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

from rote.compiler.provenance import REPORT_FILENAME, provenance_findings
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

#: ``unverified``: the tests exist but were not run, because no OS sandbox
#: is available (and the user did not pass ``--trust-skill``) or pytest
#: cannot be imported. Either way the deliverable is unverified, said loudly,
#: never a quiet pass.
TestStatus = Literal["passed", "failed", "unverified"]


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


def run_generated_tests(
    artifact_dir: Path | str, *, trust_skill: bool = False
) -> GeneratedTests | None:
    """Run ``<artifact_dir>/tests`` with pytest, if there is anything to run.

    By default the run happens inside an OS sandbox (:mod:`rote.compiler.sandbox`):
    no network, no reads outside the work dir and the interpreter, no exec
    of anything but the interpreter. With no sandbox available the tests
    are NOT run and the result is ``unverified``, never a pass.
    ``trust_skill`` (``--trust-skill``) runs them without a sandbox,
    with only the environment allowlist and an empty HOME between them and
    the user's machine.

    ``unverified`` (no sandbox, or no pytest) is reported in the compile
    output and the report, never treated as a pass, and never sent to the
    agent: it cannot fix the environment. pytest is a runtime dependency of
    rote-cli; it was dev-only once, and verification was silently off for
    every user, hiding a failing generated test from the repair loop.
    """
    from rote.compiler import sandbox

    root = Path(artifact_dir).resolve()
    tests_dir = root / "tests"
    if not tests_dir.is_dir() or not any(tests_dir.glob("test_*.py")):
        return None
    if not _pytest_available():
        return GeneratedTests(
            status="unverified",
            summary=(
                f"unverified: pytest not available for {sys.executable}, so the generated "
                f"tests were NOT run. Reinstall rote-cli (pytest is a dependency)"
            ),
        )
    box: sandbox.Sandbox | None = None
    if not trust_skill:
        box, why = sandbox.detect()
        if box is None:
            return GeneratedTests(
                status="unverified",
                summary=(
                    f"generated tests NOT run: no OS sandbox ({why}). Review them, then "
                    f"pass --trust-skill to run them without one"
                ),
            )
    try:
        with tempfile.TemporaryDirectory(prefix="rote-test-") as scratch_str:
            scratch = Path(scratch_str).resolve()
            home = scratch / "home"
            home.mkdir()
            # pytest walks up from the tests looking for a config file and
            # conftest.py files. Above the work dir is the user's project (or
            # their home), which is none of the tests' business and which the
            # sandbox denies reading. Pin the config, rootdir and conftest
            # boundary to what rote controls.
            ini = scratch / "pytest.ini"
            ini.write_text("[pytest]\n", encoding="utf-8")
            pytest_args = [
                "-m",
                "pytest",
                "tests",
                "-c",
                str(ini),
                "--rootdir",
                str(root),
                "--confcutdir",
                str(root),
                "-q",
                "--no-header",
                "-p",
                "no:cacheprovider",
                "--tb=short",
            ]
            env = generated_test_env(root, home)
            if box is None:
                argv = [sys.executable, *pytest_args]
            else:
                # -S: no site processing, so no .pth file (which can point
                # anywhere and run code) is read. The site-packages dirs that
                # hold pytest are passed explicitly instead.
                env["PYTHONPATH"] = os.pathsep.join([str(root), *sandbox.import_path()])
                argv, env = sandbox.wrap(
                    box, [sandbox.launch_executable(), "-S", *pytest_args], root, scratch, env
                )
            proc = subprocess.run(
                argv,
                cwd=root,
                env=env,
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
    # pytest's summary is the last line of stdout. The last line of the
    # combined text was whatever the interpreter wrote to stderr on its way
    # out: a uv-managed Python's ncurses, finding no terminfo inside the
    # sandbox, prints "using dumb terminal settings." there. stderr is the
    # summary only when pytest never got that far (a crash at startup).
    lines = (proc.stdout or "").strip().splitlines() or text.strip().splitlines()
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


@dataclass(frozen=True)
class EmitTarget:
    """The runtime a compile will emit for, so verification can emit it first."""

    runtime: str
    external_backend: str | None = None

    def __post_init__(self) -> None:
        from rote.adapters import ADAPTERS

        if self.runtime not in ADAPTERS:
            raise ValueError(
                f"Unknown runtime {self.runtime!r}. Available: {', '.join(sorted(ADAPTERS))}"
            )


def emit_findings(
    pipeline: Pipeline, artifact_dir: Path, target: EmitTarget | None
) -> list[ContractFinding]:
    """What the target adapter refuses, found by emitting into a temp dir.

    An adapter enforces rules no validator states: the python adapter
    refuses a hitl_gate, an MCP-incapable adapter refuses an mcp-only
    node, and emission is template code over an LLM-written IR that can
    fail in ways nobody anticipated. A dry emit runs exactly what
    ``rote emit`` will run, with the same extracted sources, so the repair
    loop sees the refusal while the agent can still fix it. A real
    bdr-outreach compile ($4.46) passed verification and then exited 1 at
    emit; that rule is now IR validation too, and this is what catches
    the next one.
    """
    if target is None:
        return []
    from rote.adapters import get_adapter

    adapter = get_adapter(
        target.runtime,
        external_backend=target.external_backend,
        extracted_source_dir=artifact_dir,
    )
    with tempfile.TemporaryDirectory(prefix="rote-dry-emit-") as out:
        try:
            adapter.emit(pipeline, Path(out) / "runtime")
        except Exception as e:  # noqa: BLE001 (every refusal is the agent's to fix)
            return [
                ContractFinding(
                    node_id="(pipeline)",
                    kind="emit_refused",
                    severity="error",
                    message=(
                        f"`rote emit --runtime {target.runtime}` refuses this "
                        f"pipeline: {type(e).__name__}: {e}"
                    ),
                    path="pipeline.yaml",
                )
            ]
    return []


#: The rubric's write-once rules, restated in every repair turn. A resumed
#: conversation is far from the rubric, and a real ops-report repair round
#: spent three turns on separate Edits to compile-report.md, each one
#: re-reading the whole context, though SKILL.md says to write it once.
WRITE_ONCE_RULES = (
    "Write-once rules still hold in a repair round: leave compile-report.md "
    "alone unless a problem below names it or a decision it records changed, "
    "and then rewrite it with ONE Write after your other fixes, never a series "
    "of Edits. Do not touch progress.ndjson. Send independent fixes as parallel "
    "tool calls in one turn."
)


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
                f"the errors name, then end your turn. {WRITE_ONCE_RULES}\n\n"
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
            f"{WRITE_ONCE_RULES}"
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
    trust_skill: bool = False,
    emit_target: EmitTarget | None = None,
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
    contains (see :mod:`rote.compiler.provenance`).

    ``emit_target`` is the runtime the compile emits for. With it, the
    deliverable must also survive that adapter's emit (:func:`emit_findings`).
    """
    try:
        return _verify_or_raise(
            artifact_dir, base_dir, run_tests, skill_dir, trust_skill, emit_target
        )
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
    trust_skill: bool,
    emit_target: EmitTarget | None,
) -> Verification:
    skill = Path(skill_dir) if skill_dir is not None else None
    if base_dir is None:
        return _verify(
            Path(artifact_dir),
            run_tests=run_tests,
            skill_dir=skill,
            trust_skill=trust_skill,
            emit_target=emit_target,
        )
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
        return _verify(
            overlay,
            run_tests=run_tests,
            skill_dir=skill,
            trust_skill=trust_skill,
            emit_target=emit_target,
        )


def _verify(
    root: Path,
    *,
    run_tests: bool,
    skill_dir: Path | None,
    trust_skill: bool,
    emit_target: EmitTarget | None,
) -> Verification:
    try:
        pipeline = load_pipeline(root / "pipeline.yaml")
    except Exception as e:  # noqa: BLE001 (any load failure is the agent's to fix)
        return Verification(pipeline=None, pipeline_error=str(e))
    implementations = implementation_status(pipeline, root)
    findings = (
        check_contracts(pipeline, root)
        + _deliverable_findings(pipeline, root, implementations)
        + provenance_findings(pipeline, root, skill_dir)
        + emit_findings(pipeline, root, emit_target)
    )
    tests = run_generated_tests(root, trust_skill=trust_skill) if run_tests else None
    return Verification(
        pipeline=pipeline,
        findings=findings,
        implementations=implementations,
        tests=tests,
    )
