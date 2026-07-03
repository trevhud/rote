"""Token counter selection and the API-backed counter (client stubbed)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from rote.eval.tokens import ApiTokenCounter, HeuristicTokenCounter, pick_token_counter


@dataclass
class _FakeCount:
    input_tokens: int


class _FakeMessages:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def count_tokens(self, **kwargs: Any) -> _FakeCount:
        self.calls.append(kwargs)
        return _FakeCount(input_tokens=1234)


class _FakeClient:
    def __init__(self) -> None:
        self.messages = _FakeMessages()


def test_heuristic_counter_scales_with_length() -> None:
    counter = HeuristicTokenCounter()
    assert counter.count("") == 0
    assert counter.count("x" * 380) == 100
    assert "approximation" in counter.method


def test_api_counter_calls_endpoint_with_model() -> None:
    client = _FakeClient()
    counter = ApiTokenCounter("claude-haiku-4-5", client=client)
    assert counter.count("hello world") == 1234
    (call,) = client.messages.calls
    assert call["model"] == "claude-haiku-4-5"
    assert call["messages"] == [{"role": "user", "content": "hello world"}]
    assert "count_tokens" in counter.method


def test_api_counter_skips_call_for_empty_text() -> None:
    client = _FakeClient()
    assert ApiTokenCounter("m", client=client).count("") == 0
    assert client.messages.calls == []


def test_pick_prefers_api_when_key_and_model_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    counter = pick_token_counter(model="claude-haiku-4-5")
    assert isinstance(counter, ApiTokenCounter)


def test_pick_uses_heuristic_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert isinstance(pick_token_counter(model="claude-haiku-4-5"), HeuristicTokenCounter)


def test_pick_uses_heuristic_without_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert isinstance(pick_token_counter(model=None), HeuristicTokenCounter)
