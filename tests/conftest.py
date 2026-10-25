"""Shared fixtures for the rote test suite."""

from __future__ import annotations

import ipaddress
import os
import shutil
import socket
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

from rote import extras as rote_extras
from rote import isolation
from rote.ir import Pipeline, load_pipeline
from tests import _inference_guard
from tests import _node_trees as node_trees

REPO_ROOT = Path(__file__).resolve().parent.parent
BDR_PIPELINE_YAML = REPO_ROOT / "examples" / "bdr-outreach" / "expected" / "pipeline.yaml"
EXAMPLES_DIR = REPO_ROOT / "examples"

# ───────── The committed examples/ tree is read-only to the suite ─────────
#
# Five adapter suites used to emit straight into
# examples/bdr-outreach/expected/runtimes/<runtime>/, three of them after
# an rmtree. Emitting without the example's extracted/ sources replaced
# the committed real implementations with stubs, so running pytest left a
# dirty tree, and on the merge-queue worktree that blocked merges. Tests
# emit into tmp_path and diff against the snapshot instead
# (tests/test_committed_snapshots.py). This guard fails the session if
# anything under examples/ changed, so the pattern cannot come back.

#: Install and import artifacts, never committed, created by tooling
#: rather than by the code under test.
_EXAMPLES_IGNORED_PARTS = frozenset({"__pycache__", "node_modules"})


def examples_fingerprint(root: Path) -> dict[str, str]:
    """``{relative path: sha256}`` for every file under ``root``."""
    import hashlib

    prints: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or _EXAMPLES_IGNORED_PARTS.intersection(path.parts):
            continue
        prints[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return prints


def fingerprint_changes(before: dict[str, str], after: dict[str, str]) -> list[str]:
    """Human-readable lines for every file added, removed, or modified."""
    lines = [f"added: {p}" for p in sorted(after.keys() - before.keys())]
    lines += [f"removed: {p}" for p in sorted(before.keys() - after.keys())]
    lines += [
        f"modified: {p}" for p in sorted(before.keys() & after.keys()) if before[p] != after[p]
    ]
    return lines


# ───────── No inference, no real per-user state: once, for the session ─────────
#
# Applied to os.environ itself before anything is collected, so neither a
# per-test ``monkeypatch.undo()`` nor a child process escapes it (see
# tests/_inference_guard.py for the two leaks that made this necessary, one
# of which billed a real compile). Per-test fixtures below still move the
# state paths under each test's own tmp_path; undoing them falls back here,
# never to the developer's real files.

#: The session's state dir. The environment from before the session is
#: deliberately kept nowhere: a test that could read it back could restore
#: the real PATH.
SESSION_STATE = pytest.StashKey[Path]()


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config: pytest.Config) -> None:
    state = Path(tempfile.mkdtemp(prefix="rote-test-session-")).resolve()
    config.stash[SESSION_STATE] = state
    _inference_guard.isolate(os.environ, state)
    reachable = _inference_guard.reachable_clis(os.environ)
    if reachable:
        pytest.exit(
            "the suite could not take the subscription CLIs off PATH: " + "; ".join(reachable),
            returncode=pytest.ExitCode.USAGE_ERROR,
        )


def pytest_unconfigure(config: pytest.Config) -> None:
    state = config.stash.get(SESSION_STATE, None)
    if state is not None:
        _inference_guard.release_mirrors(state)
        shutil.rmtree(state)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item: pytest.Item) -> Iterator[None]:
    """Fail any test during which a bare ``claude`` or ``codex`` resolves.

    Checked with the test's fixtures in place, before and after its body,
    both in-process and from a child shell, because a child process is
    where the $0.29 leak happened. A failure, never a skip: a skipped
    check is how spending goes unnoticed.
    """
    _require_no_subscription_cli("before the test body ran")
    result = yield
    _require_no_subscription_cli("after the test body ran")
    return result


def _require_no_subscription_cli(when: str) -> None:
    reachable = _inference_guard.reachable_clis(os.environ)
    if reachable:
        pytest.fail(
            f"a subscription CLI is reachable {when} ({'; '.join(reachable)}). "
            "A test may run a stub by absolute path, never a bare name on PATH.",
            pytrace=False,
        )


# ───────── The suite needs the dev extra ─────────
#
# Tests used to guard optional dependencies one by one with importorskip.
# The guards kept missing cases, and where they held they skipped
# silently: a reviewer's run passed while skipping every core test of the
# skill runtime. The suite now runs only where CI runs it, with the dev
# extra installed, and says so before collecting anything.

#: What CI runs, and so the one fix for a missing dev requirement.
DEV_INSTALL = "uv sync --extra dev"


def missing_dev_requirements() -> list[str]:
    """Each requirement of pyproject's dev extra this environment does not meet.

    Checked by distribution, not by import: fastmcp's server half and the
    slim client every base install has are both ``import fastmcp``, and
    types-pyyaml has no module at all."""
    import tomllib
    from importlib import metadata

    from packaging.requirements import Requirement

    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    missing = []
    for spec in project["project"]["optional-dependencies"]["dev"]:
        requirement = Requirement(spec)
        if requirement.marker is not None and not requirement.marker.evaluate():
            continue
        try:
            version = metadata.version(requirement.name)
        except metadata.PackageNotFoundError:
            missing.append(spec)
            continue
        if not requirement.specifier.contains(version, prereleases=True):
            missing.append(f"{spec} (found {version})")
    return missing


def pytest_sessionstart(session: pytest.Session) -> None:
    missing = missing_dev_requirements()
    if missing:
        pytest.exit(
            "the test suite needs the dev extra; missing: "
            f"{', '.join(missing)}. Install it with: {DEV_INSTALL}",
            returncode=pytest.ExitCode.USAGE_ERROR,
        )
    session.config.stash[_EXAMPLES_BEFORE] = examples_fingerprint(EXAMPLES_DIR)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    before = session.config.stash.get(_EXAMPLES_BEFORE, None)
    if before is None:
        return
    changes = fingerprint_changes(before, examples_fingerprint(EXAMPLES_DIR))
    if not changes:
        return
    session.config.stash[_EXAMPLES_CHANGES] = changes
    session.exitstatus = pytest.ExitCode.TESTS_FAILED


def pytest_terminal_summary(terminalreporter: pytest.TerminalReporter) -> None:
    changes = terminalreporter.config.stash.get(_EXAMPLES_CHANGES, None)
    if not changes:
        return
    terminalreporter.section("committed examples/ changed during the run", red=True)
    terminalreporter.line(
        "A test wrote into the committed examples/ tree. Emit into tmp_path instead "
        "(compare against the snapshot from there). Restore with: git checkout -- examples/"
    )
    for line in changes:
        terminalreporter.line(f"  {line}")


_EXAMPLES_BEFORE = pytest.StashKey[dict[str, str]]()
_EXAMPLES_CHANGES = pytest.StashKey[list[str]]()


@pytest.fixture(autouse=True)
def _isolated_mcp_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No test may read (or write!) the developer's real MCP registry,
    token store, or app registry — several code paths (eval MCP wiring,
    emitted-helper resolution, the `rote mcp` commands, and `rote emit`
    recording the output dir) consult them by default."""
    # Every per-user state path (rote.isolation.STATE_PATHS), including the
    # rote-cloud credential, which a developer's real login must never
    # satisfy or be clobbered by, and the layered config, which must not
    # change what any command resolves. Each points under tmp_path.
    # The session already moved them all (and dropped the cloud credential
    # and the config env twins); this gives each test its own copy.
    for key, value in isolation.isolated_env(tmp_path, base={}).items():
        monkeypatch.setenv(key, value)
    # `rote compile` now always constructs its JSONL progress sink (the
    # <out>/progress.jsonl sidecar), whose price enrichment does a live
    # catalog fetch. No unit test may touch the network for it — tests
    # that exercise pricing re-patch this same seam with real numbers.
    monkeypatch.setattr(
        "rote.cli._JsonlProgressSink._resolve_prices",
        staticmethod(lambda _model_id: None),
    )
    # `rote run`'s judge-usage line prices rows the same way.
    monkeypatch.setattr("rote.cli._judge_price_catalog", _offline_price_catalog)
    # And the api driver prices its own run (cost_usd in its metadata): no
    # test may reach the live catalog or read the developer's cached copy.
    monkeypatch.setattr("rote.eval.pricing._get_json", _offline_price_fetch)
    # Vendor specs (--backend api) are fetched from the vendor. Tests use
    # the trimmed real documents in tests/fixtures/vendor_specs through an
    # injected fetch, never the network, and never the developer's cache
    # (ROTE_SPEC_CACHE_DIR above).
    monkeypatch.setattr("rote.vendor_specs.store.http_get_json", _offline_spec_fetch)
    # Nor install an app's vendor libraries (rote.vendor_specs.deps).
    monkeypatch.setattr("rote.vendor_specs.deps._run_uv", _no_package_install)


def _no_package_install(args: list[str]) -> None:
    from rote.vendor_specs.deps import VendorDepsError

    raise VendorDepsError(f"tests never install packages ({' '.join(args)})")


def _offline_price_fetch(*args: object, **kwargs: object) -> object:
    from rote.eval.pricing import PricingError

    raise PricingError("tests never fetch live prices")


def _offline_spec_fetch(url: str) -> object:
    from rote.vendor_specs.store import SpecUnavailable

    raise SpecUnavailable(f"tests never fetch vendor specs ({url})")


def _offline_price_catalog() -> object:
    from rote.eval.pricing import PricingError

    raise PricingError("tests never fetch live prices")


class NetworkBlocked(RuntimeError):
    """A fast test tried to reach a host that is not this machine."""


def _loopback(address: object) -> bool:
    """True for an AF_UNIX path or an address on this machine."""
    if not isinstance(address, tuple) or not address:
        return True  # AF_UNIX socket paths, and anything that is not an IP pair
    host = address[0]
    if not isinstance(host, str):
        return True
    if host in ("localhost", ""):
        return True
    try:
        ip = ipaddress.ip_address(host.split("%", 1)[0])
    except ValueError:
        return False  # an unresolved hostname
    return ip.is_loopback or ip.is_unspecified


@pytest.fixture(autouse=True)
def _no_network_in_fast_tests(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fast test may only connect to this machine.

    Found the hard way (#150 review): with one credential check mutated
    away, a Clef decider test made real, retried HTTPS calls to
    api.cloudflare.com for 84 seconds, and every assertion still ran.
    Loopback stays open for the fixture servers (fastmcp, local Messages
    stubs, DBOS on localhost). Tests marked ``slow`` are exempt: they are
    outside the default run and some legitimately download (npm install,
    the inngest-cli binary). Subprocesses are not covered; this guards
    in-process clients (urllib, httpx, SDKs), which is where a test's own
    code calls out.
    """
    if request.node.get_closest_marker("slow") is not None:
        return
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex

    def connect(self: socket.socket, address: object) -> None:
        if not _loopback(address):
            raise NetworkBlocked(f"fast tests may not connect to {address!r}; stub the call")
        real_connect(self, address)  # type: ignore[arg-type]

    def connect_ex(self: socket.socket, address: object) -> int:
        if not _loopback(address):
            raise NetworkBlocked(f"fast tests may not connect to {address!r}; stub the call")
        return real_connect_ex(self, address)  # type: ignore[arg-type]

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)


#: rote.extras' real ``direct_url.json`` reader, before the fixture below
#: hides it (tests/test_install_extra.py checks it against this checkout).
REAL_DIRECT_URL_SOURCE = rote_extras._direct_url_source
REAL_HAS_PIP = rote_extras._has_pip


@pytest.fixture(autouse=True)
def _released_install_and_no_installs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests see rote as a user does, and never install a package.

    The suite runs from an editable checkout in a uv venv without pip, so
    ``rote.extras`` would derive ``uv pip install -e <checkout>[...]``
    hints. Tests assert the hints a PyPI install in a pip venv prints, so
    the checkout's ``direct_url.json`` and pip's absence are hidden
    (the uv-tool and uvx contexts come from ``sys.prefix`` and stay
    testable). And no test may run an installer: one that approves an
    install plan would otherwise change the suite's own environment. A test
    of the install flow patches ``_run_installer`` with its own fake.
    """
    import rote.setup.install_extra

    monkeypatch.setattr(rote_extras, "_direct_url_source", lambda: None)
    # The suite's venv is uv's, without pip; a user's venv usually has it.
    monkeypatch.setattr(rote_extras, "_has_pip", lambda: True)

    def _no_installs(argv: object) -> None:
        raise AssertionError(f"a test tried to install packages: {argv!r}")

    monkeypatch.setattr(rote.setup.install_extra, "_run_installer", _no_installs)


@pytest.fixture(scope="session", autouse=True)
def _node_trees_root(tmp_path_factory: pytest.TempPathFactory) -> None:
    """Where ``tests/_node_trees.py`` keeps the session's shared installs."""
    node_trees.set_root(tmp_path_factory.getbasetemp() / "node-trees")


@pytest.fixture(autouse=True)
def _shared_node_trees_in_runners(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Slow tests that ``rote run`` an emitted TS app share its node_modules.

    The runners install through ``ensure_npm_install``. Routing their copies
    through ``node_trees.install`` gives them the same per-dependency-set
    trees as the tests that install directly, and a miss still runs the real
    ``ensure_npm_install``. Fast tests keep the real function, because some
    of them assert on what it does.
    """
    if request.node.get_closest_marker("slow") is None:
        return
    for module in ("_node", "cloudflare", "dbos_ts", "inngest"):
        monkeypatch.setattr(f"rote.runners.{module}.ensure_npm_install", node_trees.install)


@pytest.fixture(scope="session")
def bdr_pipeline() -> Pipeline:
    """The canonical BDR pipeline — the IR that exercises all five node kinds.

    Session-scoped: one parse serves the whole suite. Tests must treat
    it as immutable; anything that needs to mutate should deep-copy (see
    test_dbos_adapter's copy.deepcopy usage) or build its own pipeline
    via tests._helpers.mini_pipeline.
    """
    assert BDR_PIPELINE_YAML.exists(), f"Missing fixture: {BDR_PIPELINE_YAML}"
    return load_pipeline(BDR_PIPELINE_YAML)
