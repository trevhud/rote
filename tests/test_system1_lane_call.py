"""System-1's call site retries a transient failure on the same lane, and nothing else."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from rote.compiler.system1 import lane_call
from rote.inference import _runtime_helper as helper

ANSWER = ({"ok": True}, [("m", {"input_tokens": 1})])


class _Flaky:
    """Stands in for the helper's structured_call: fails with ``errors`` in turn, then answers."""

    def __init__(self, *errors: str) -> None:
        self.errors = list(errors)
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.errors:
            raise RuntimeError(self.errors.pop(0))
        return ANSWER


@pytest.fixture
def waits(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    slept: list[float] = []
    monkeypatch.setattr(lane_call, "_sleep", slept.append)
    return slept


def _call(monkeypatch: pytest.MonkeyPatch, flaky: _Flaky, **over: Any) -> Any:
    monkeypatch.setattr(helper, "structured_call", flaky)
    kwargs = {"node_id": "system1_structure", "provider": "api", "client": "openai", "model": "m"}
    return lane_call.structured_call(**{**kwargs, "prompt": "p", "output_schema": {}, **over})


@pytest.mark.parametrize(
    "error",
    [
        "APIStatusError (HTTP 408) calling @cf/zai-org/glm-5.2 for system1_structure via api: "
        "AiError: Request timeout",
        "RateLimitError (HTTP 429) calling m for n via api: slow down",
        "InternalServerError (HTTP 529) calling m for n via api: overloaded",
        "APIConnectionError calling m for n via api: Connection error.",
        "claude -p failed for system1_structure: API Error: 503 upstream",
    ],
)
def test_a_transient_failure_is_retried_on_the_same_lane(
    error: str, monkeypatch: pytest.MonkeyPatch, waits: list[float]
) -> None:
    flaky = _Flaky(error, error)
    assert _call(monkeypatch, flaky) == ANSWER
    assert len(flaky.calls) == 3 and waits == [2.0, 8.0]
    # Same lane, same model, same request every time: a retry, not a fallback.
    assert all(c == flaky.calls[0] for c in flaky.calls)


@pytest.mark.parametrize(
    "error",
    [
        "APIStatusError (HTTP 402) calling m for n via api: Insufficient balance",
        "BadRequestError (HTTP 400) calling m for n via api: bad schema",
        "AuthenticationError (HTTP 401) calling m for n via api: no key",
        "claude -p failed for n: API Error: 400 invalid_request_error",
    ],
)
def test_a_permanent_failure_is_raised_at_once(
    error: str, monkeypatch: pytest.MonkeyPatch, waits: list[float]
) -> None:
    flaky = _Flaky(error)
    with pytest.raises(RuntimeError) as e:
        _call(monkeypatch, flaky)
    assert str(e.value) == error and len(flaky.calls) == 1 and waits == []


def test_retries_are_bounded_and_the_real_error_surfaces(
    monkeypatch: pytest.MonkeyPatch, waits: list[float]
) -> None:
    error = "APIStatusError (HTTP 408) calling m for n via api: Request timeout"
    flaky = _Flaky(*[error] * 10)
    with pytest.raises(RuntimeError, match=r"HTTP 408.*Request timeout \(after 3 attempts\)"):
        _call(monkeypatch, flaky)
    assert len(flaky.calls) == lane_call.ATTEMPTS == 3


def test_a_codex_lane_is_retried_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, waits: list[float]
) -> None:
    """The codex transport goes through the same retry."""
    attempts: list[int] = []

    def flaky_codex(**kwargs: Any) -> Any:
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError('codex exec for n failed (exit 1): {"message": "Connection reset"}')
        return ANSWER

    monkeypatch.setattr(lane_call, "call", flaky_codex)
    out = _call(monkeypatch, _Flaky(), provider=lane_call.PROVIDER, sdk_client=lane_call.CodexCli())
    assert out == ANSWER and len(attempts) == 2 and waits == [2.0]


def test_the_drivers_client_does_not_retry_under_ours(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, waits: list[float]
) -> None:
    """A gateway that keeps timing out gets 3 requests, not 3 x the SDK's 3,
    and each still goes to the driver's own endpoint."""
    pytest.importorskip("openai")
    from rote.compiler.system1 import tiebreak
    from tests._helpers import GATEWAY_URL, openai_compile_driver, openai_gateway

    requests = openai_gateway(monkeypatch, [408] * 10)
    lane = tiebreak.lane_for(openai_compile_driver())
    with pytest.raises(RuntimeError, match=r"HTTP 408.*\(after 3 attempts\)"):
        lane_call.structured_call(
            node_id="system1_structure",
            provider=lane.provider,
            client=lane.client,
            model=lane.model,
            prompt="p",
            output_schema={"type": "object", "properties": {}, "additionalProperties": False},
            sdk_client=lane.sdk_client,
        )
    assert len(requests) == 3 and waits == [2.0, 8.0]
    assert all(str(r.url) == f"{GATEWAY_URL}/chat/completions" for r in requests)
    # The lane keeps the driver's client as it was; only System-1's calls opt out.
    assert lane.sdk_client.max_retries == 2


def test_a_permanent_status_quoting_a_transient_one_is_not_retried(
    monkeypatch: pytest.MonkeyPatch, waits: list[float]
) -> None:
    """The match is anchored on the status the lane reports, not the body."""
    error = (
        "BadRequestError (HTTP 400) calling m for n via api: schema rejected; "
        "see HTTP 429 limits and API Error: 503 docs"
    )
    flaky = _Flaky(error)
    with pytest.raises(RuntimeError):
        _call(monkeypatch, flaky)
    assert len(flaky.calls) == 1 and waits == []
    assert not lane_call.transient(
        RuntimeError("claude -p failed for n: bad input, not API Error: 529")
    )


class _Billed:
    """Fails like a billed-then-failed call (usage on the error), then answers."""

    def __init__(self, *errors: str) -> None:
        self.errors = list(errors)

    def __call__(self, **kwargs: Any) -> Any:
        if self.errors:
            e = RuntimeError(self.errors.pop(0))
            e.usage_rows = [("m", {"input_tokens": 7, "output_tokens": 3})]  # type: ignore[attr-defined]
            raise e
        return ANSWER


def test_a_billed_failed_attempt_is_counted(
    monkeypatch: pytest.MonkeyPatch, waits: list[float]
) -> None:
    transient = "claude -p failed for n: API Error: 529 overloaded"
    _, rows = _call(monkeypatch, _Billed(transient))  # type: ignore[arg-type]
    assert rows == [("m", {"input_tokens": 7, "output_tokens": 3}), *ANSWER[1]]
    # A final failure carries every billed attempt's usage.
    with pytest.raises(RuntimeError) as e:
        _call(monkeypatch, _Billed(*[transient] * 3))  # type: ignore[arg-type]
    assert len(e.value.usage_rows) == 3  # type: ignore[attr-defined]
    with pytest.raises(RuntimeError) as e:
        _call(monkeypatch, _Billed("claude -p failed for n: model not found"))  # type: ignore[arg-type]
    assert len(e.value.usage_rows) == 1  # type: ignore[attr-defined]
