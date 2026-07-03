"""Token counting for the estimator.

Two counters, chosen explicitly and *named in the scorecard output* so
a reader always knows which produced the numbers:

* :class:`ApiTokenCounter` — Anthropic's free ``count_tokens`` endpoint,
  exact for the given model. Used when an API key is available.
* :class:`HeuristicTokenCounter` — a chars-per-token approximation.
  Anthropic doesn't publish its tokenizer, so offline counting can only
  approximate; the method label makes that honest.

The choice is a visible parameter of the estimate, not a silent
fallback chain.
"""

from __future__ import annotations

import os
from typing import Any, Protocol

from rote.eval.priors import Priors


class TokenCounter(Protocol):
    """Counts tokens in text; ``method`` names how, for the scorecard."""

    @property
    def method(self) -> str: ...

    def count(self, text: str) -> int: ...


class HeuristicTokenCounter:
    """Chars-per-token approximation (see ``Priors.chars_per_token``)."""

    def __init__(self, priors: Priors | None = None) -> None:
        self._chars_per_token = (priors or Priors()).chars_per_token

    @property
    def method(self) -> str:
        return f"chars/{self._chars_per_token:g} approximation"

    def count(self, text: str) -> int:
        if not text:
            return 0
        return max(1, round(len(text) / self._chars_per_token))


class ApiTokenCounter:
    """Exact counting via Anthropic's free ``count_tokens`` endpoint.

    ``POST /v1/messages/count_tokens`` takes a model id plus messages
    and returns ``{input_tokens}`` — no generation, no charge. A model
    id is required because counting is tokenizer-specific; callers pass
    one from the live catalog rather than hardcoding.

    Failures raise: a set-but-broken ``ANTHROPIC_API_KEY`` should be
    fixed (or unset, which selects the heuristic), not silently papered
    over with an approximation labeled as exact.
    """

    def __init__(self, model: str, client: Any | None = None) -> None:
        if client is None:
            from anthropic import Anthropic

            client = Anthropic()
        self._client = client
        self._model = model

    @property
    def method(self) -> str:
        return f"anthropic count_tokens ({self._model})"

    def count(self, text: str) -> int:
        if not text:
            return 0
        result = self._client.messages.count_tokens(
            model=self._model,
            messages=[{"role": "user", "content": text}],
        )
        return int(result.input_tokens)


def pick_token_counter(priors: Priors | None = None, model: str | None = None) -> TokenCounter:
    """Choose the best counter available in this environment.

    Exact API counting (Anthropic ``count_tokens``, free) when an API
    key is present and a model id is known; the labeled heuristic
    otherwise. The scorecard reports whichever was used via ``method``.
    """
    if model and os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return ApiTokenCounter(model)
        except ImportError:
            pass  # anthropic extra not installed; the heuristic labels itself
    return HeuristicTokenCounter(priors)
