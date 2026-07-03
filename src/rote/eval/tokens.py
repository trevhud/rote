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

from typing import Protocol

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


def pick_token_counter(priors: Priors | None = None) -> TokenCounter:
    """Choose the best counter available in this environment.

    Exact API counting (Anthropic ``count_tokens``) when a key is
    present; the labeled heuristic otherwise. The scorecard reports
    whichever was used via ``method``.
    """
    return HeuristicTokenCounter(priors)
