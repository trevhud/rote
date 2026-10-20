"""Shared fixtures for the rote test suite."""

from __future__ import annotations

import ipaddress
import os
import shutil
import socket
from pathlib import Path

import pytest

from rote.ir import Pipeline, load_pipeline

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


def pytest_sessionstart(session: pytest.Session) -> None:
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
    monkeypatch.setenv("ROTE_MCP_CONFIG", str(tmp_path / "mcp-config" / "mcp.json"))
    monkeypatch.setenv("ROTE_MCP_TOKEN_DIR", str(tmp_path / "mcp-tokens"))
    monkeypatch.setenv("ROTE_APPS_PATH", str(tmp_path / "rote-apps" / "apps.json"))
    # Same rule for the rote-cloud login: a developer's real credential
    # must never satisfy (or be clobbered by) a test's resolution chain.
    monkeypatch.setenv("ROTE_CLOUD_CRED_PATH", str(tmp_path / "cloud-cred" / "cloud.json"))
    # And for the layered config: a developer's ~/.config/rote/config.yaml
    # (or a stray rote.yaml above the test cwd) must not change what any
    # command resolves. Both paths point at files that don't exist.
    monkeypatch.setenv("ROTE_CONFIG_PATH", str(tmp_path / "rote-config" / "config.yaml"))
    monkeypatch.setenv("ROTE_PROJECT_CONFIG_PATH", str(tmp_path / "rote-config" / "rote.yaml"))
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
    monkeypatch.setattr(
        "rote.eval.pricing._default_cache_path", lambda: tmp_path / "pricing-cache.json"
    )
    # Vendor specs (--backend api) are fetched from the vendor. Tests use
    # the trimmed real documents in tests/fixtures/vendor_specs through an
    # injected fetch, never the network, and never the developer's cache.
    monkeypatch.setenv("ROTE_SPEC_CACHE_DIR", str(tmp_path / "vendor-specs"))
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


#: Credentials that would let a test reach a real model. Scrubbed from
#: every test's environment — see :func:`_no_inference_in_tests`.
_INFERENCE_CREDENTIAL_VARS = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "ROTE_INFERENCE",
    "ROTE_CLOUD_TOKEN",
)

#: Subscription-auth CLIs. These are the dangerous ones: `claude -p`
#: needs no API key at all, so a test that merely *finds* the binary
#: spends the developer's subscription with nothing in the environment
#: to scrub. Exactly how two DBOS e2e suites quietly burned real
#: inference on every run until it was caught by timing.
_SUBSCRIPTION_CLIS = frozenset({"claude", "codex"})


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


@pytest.fixture(autouse=True)
def _no_inference_in_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    """A test may never spend tokens.

    The project's split: anything that does not require inference is a
    **test** — automated, fast, run on every push. Anything that costs
    tokens is an **eval** — run deliberately and periodically, never
    from CI. This fixture makes the first half impossible to violate by
    accident instead of merely documented.

    Two independent leaks, both closed here:

    1. **Credentials in the environment.** A developer running the suite
       locally has real keys exported; a vendor SDK constructed without
       an explicit key picks them up silently.
    2. **A subscription CLI on PATH.** `claude -p` authenticates from an
       OAuth session, so no env var gates it — the only defense is not
       finding the binary. `shutil.which` is wrapped rather than
       replaced so that lookups the e2e suites depend on (node, npm,
       docker, wrangler) still resolve normally.

    Only a *bare name* lookup is blocked, because only that consults
    PATH and can reach the developer's real install. Several tests build
    a fake `claude` script under tmp_path and pass its absolute path as
    `executable=`; that is a stub, not a subscription, and it still
    resolves.

    A test that legitimately needs one of these monkeypatches it back:
    this fixture runs first, so a test-local patch wins.
    """
    for var in _INFERENCE_CREDENTIAL_VARS:
        monkeypatch.delenv(var, raising=False)

    real_which = shutil.which

    def _which_without_subscription_clis(cmd: str, *args: object, **kwargs: object):  # noqa: ANN202
        if isinstance(cmd, str) and os.sep not in cmd and cmd in _SUBSCRIPTION_CLIS:
            return None
        return real_which(cmd, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(shutil, "which", _which_without_subscription_clis)


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
