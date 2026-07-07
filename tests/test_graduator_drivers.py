"""Tests for the graduator driver Protocol + registry + auto-detect.

These tests cover the v0 scaffolding of the driver system. The concrete
``run()`` implementations are stubs that raise ``NotImplementedError``
— they'll be exercised in task #13's test suite.

What's covered here:

* The Protocol is importable and has the right shape.
* Every driver in ``DRIVERS`` can be instantiated without a network
  call or subprocess spawn.
* ``is_available()`` returns a tuple on every driver and respects
  environment state (env var presence, CLI presence on PATH).
* ``get_driver()`` raises a helpful error for unknown names.
* ``auto_detect()`` returns None or a driver cleanly and never
  raises on missing state.
* ``available_drivers()`` returns diagnostic triples for all drivers.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from rote.graduator.drivers import (
    AUTO_DETECT_ORDER,
    DRIVERS,
    DriverError,
    DriverResult,
    auto_detect,
    available_drivers,
    get_driver,
)
from rote.graduator.drivers.anthropic_api import AnthropicApiDriver
from rote.graduator.drivers.claude import ClaudeDriver
from rote.graduator.drivers.codex import CodexDriver

# ───────── Registry + Protocol shape ─────────


def test_all_three_drivers_registered() -> None:
    assert set(DRIVERS) == {"claude", "codex", "api"}


def test_auto_detect_order_matches_registry() -> None:
    assert set(AUTO_DETECT_ORDER) == set(DRIVERS)
    # Specifically verify the priority: claude first for the most common
    # subscription path, api last because it requires explicit opt-in.
    assert AUTO_DETECT_ORDER[0] == "claude"
    assert AUTO_DETECT_ORDER[-1] == "api"


def test_get_driver_returns_instance() -> None:
    assert isinstance(get_driver("claude"), ClaudeDriver)
    assert isinstance(get_driver("codex"), CodexDriver)
    assert isinstance(get_driver("api"), AnthropicApiDriver)


def test_get_driver_unknown_name_raises_keyerror_with_message() -> None:
    with pytest.raises(KeyError) as excinfo:
        get_driver("bogus")
    message = str(excinfo.value)
    assert "bogus" in message
    assert "claude" in message  # helpful list of available drivers
    assert "codex" in message
    assert "api" in message


def test_driver_instances_expose_name_attribute() -> None:
    assert get_driver("claude").name == "claude"
    assert get_driver("codex").name == "codex"
    assert get_driver("api").name == "api"


def test_drivers_satisfy_protocol_at_runtime() -> None:
    """Each driver exposes the Protocol's methods.

    Protocols aren't checked by isinstance by default, but we can
    assert the methods exist and are callable.
    """
    for name in DRIVERS:
        driver = get_driver(name)
        assert hasattr(driver, "name")
        assert callable(driver.is_available)
        assert callable(driver.run)


# ───────── is_available() behavior ─────────


def test_claude_driver_is_available_when_cli_present() -> None:
    """With ``shutil.which("claude")`` patched to return a path,
    the driver reports available."""
    with patch("rote.graduator.drivers.claude.which", return_value="/usr/local/bin/claude"):
        driver = ClaudeDriver()
        available, reason = driver.is_available()
        assert available is True
        assert reason == ""


def test_claude_driver_reports_install_message_when_cli_missing() -> None:
    with patch("rote.graduator.drivers.claude.which", return_value=None):
        driver = ClaudeDriver()
        available, reason = driver.is_available()
        assert available is False
        assert "claude" in reason.lower()
        assert "install" in reason.lower()


def test_codex_driver_is_available_when_cli_present() -> None:
    with patch("rote.graduator.drivers.codex.which", return_value="/usr/local/bin/codex"):
        driver = CodexDriver()
        available, reason = driver.is_available()
        assert available is True


def test_codex_driver_reports_install_message_when_cli_missing() -> None:
    with patch("rote.graduator.drivers.codex.which", return_value=None):
        driver = CodexDriver()
        available, reason = driver.is_available()
        assert available is False
        assert "codex" in reason.lower()
        assert "install" in reason.lower()


def test_api_driver_reports_missing_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    driver = AnthropicApiDriver()
    available, reason = driver.is_available()
    assert available is False
    assert "ANTHROPIC_API_KEY" in reason


def test_api_driver_is_available_when_env_var_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    driver = AnthropicApiDriver()
    # Only true if the anthropic package is installed; in our dev env
    # it is (see pyproject.toml [dev] extras), so this should pass.
    available, reason = driver.is_available()
    assert available is True, f"expected available, got reason: {reason}"


def test_api_driver_reports_missing_package_when_anthropic_not_importable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Simulate the case where the `anthropic` extra wasn't installed."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with patch("rote.graduator.drivers.anthropic_api._ANTHROPIC_AVAILABLE", False):
        driver = AnthropicApiDriver()
        available, reason = driver.is_available()
        assert available is False
        assert "rote[api]" in reason


# ───────── auto_detect() behavior ─────────


def test_auto_detect_returns_first_available() -> None:
    """With claude available (mocked) and others unavailable, pick claude."""
    with (
        patch("rote.graduator.drivers.claude.which", return_value="/usr/local/bin/claude"),
        patch("rote.graduator.drivers.codex.which", return_value=None),
    ):
        driver = auto_detect()
        assert driver is not None
        assert driver.name == "claude"


def test_auto_detect_falls_through_to_codex_when_claude_missing() -> None:
    with (
        patch("rote.graduator.drivers.claude.which", return_value=None),
        patch("rote.graduator.drivers.codex.which", return_value="/usr/local/bin/codex"),
    ):
        driver = auto_detect()
        assert driver is not None
        assert driver.name == "codex"


def test_auto_detect_returns_none_when_no_driver_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with (
        patch("rote.graduator.drivers.claude.which", return_value=None),
        patch("rote.graduator.drivers.codex.which", return_value=None),
    ):
        driver = auto_detect()
        assert driver is None


def test_available_drivers_returns_triples_for_all(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Diagnostic function returns status for every registered driver."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with (
        patch("rote.graduator.drivers.claude.which", return_value=None),
        patch("rote.graduator.drivers.codex.which", return_value=None),
    ):
        triples = available_drivers()

    assert len(triples) == 3
    names = [t[0] for t in triples]
    assert names == list(AUTO_DETECT_ORDER)
    # All should report unavailable with a helpful reason
    for name, available, reason in triples:
        assert available is False, f"{name} unexpectedly available"
        assert reason, f"{name} has no reason when unavailable"


# ───────── run() implementations ─────────


# All three drivers' run() methods are implemented and covered by their
# own dedicated test modules:
#   * ClaudeDriver    → test_claude_driver.py
#   * CodexDriver     → test_codex_driver.py
#   * AnthropicApiDriver → test_anthropic_driver.py
# (tool dispatch, path security, subprocess arg/env shape, missing
# pipeline.yaml, nonzero-exit recovery, metadata). No stub assertions
# remain here.


# ───────── Shared types ─────────


def test_driver_error_carries_details() -> None:
    err = DriverError("something broke", details="stderr output here")
    assert str(err) == "something broke"
    assert err.details == "stderr output here"


def test_driver_result_is_constructible() -> None:
    result = DriverResult(
        pipeline_yaml_path=Path("/tmp/pipeline.yaml"),
        work_dir=Path("/tmp/work"),
        driver_name="claude",
        metadata={"tokens": 1234},
    )
    assert result.driver_name == "claude"
    assert result.metadata == {"tokens": 1234}
