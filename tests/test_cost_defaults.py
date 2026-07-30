"""Defaults that cost real money when they change.

Two compiler-driver constants have burned this project before, and both
are documented in CLAUDE.md as "don't change this" — but documentation
is not a test. A mutation sweep confirmed it: flipping the default model
to Opus and halving the turn budget both left the entire suite green.

The existing driver tests compare against the constants *symbolically*
(``args[args.index("--model") + 1] == DEFAULT_MODEL``), which is right
for checking the flag is wired up and useless for catching a change to
the constant itself — both sides move together.

These tests pin the *property* rather than the literal, so a legitimate
upgrade (sonnet-4-6 → sonnet-5) passes while the expensive regression
fails.
"""

from __future__ import annotations

import re

from rote.compiler.drivers.anthropic_api import DEFAULT_MODEL as API_DEFAULT_MODEL
from rote.compiler.drivers.claude import DEFAULT_MAX_TURNS
from rote.compiler.drivers.claude import DEFAULT_MODEL as CLI_DEFAULT_MODEL

#: Compiling BDR ran ~$3.50 per attempt on Opus and exhausted a Claude
#: Max "extra usage" budget in two runs. Sonnet follows the structured
#: rubric perfectly well; Opus is ~5x the price for no measured gain.
_OPUS_RE = re.compile(r"opus", re.IGNORECASE)

#: BDR-scale skills need ~25 tool calls minimum and realistically 40-50
#: with exploration. The original default of 30 produced a hard
#: `error_max_turns` failure on the first real run.
_MIN_MAX_TURNS = 60


def test_compiler_drivers_do_not_default_to_opus() -> None:
    """Both drivers default to a non-Opus model.

    Asserted as "not Opus" rather than an exact model id so bumping the
    Sonnet generation stays a one-line change, while the regression this
    exists to prevent still fails.
    """
    for name, model in (
        ("ClaudeDriver", CLI_DEFAULT_MODEL),
        ("AnthropicApiDriver", API_DEFAULT_MODEL),
    ):
        assert not _OPUS_RE.search(model), (
            f"{name}.DEFAULT_MODEL is {model!r}. Opus is ~5x Sonnet's price "
            f"and Sonnet follows the compiler rubric fine — two Opus runs of "
            f"BDR exhausted a Max extra-usage budget. If you have specific "
            f"evidence a skill needs Opus, pass model= explicitly rather than "
            f"changing the default for everyone."
        )


def test_both_drivers_share_a_default_model() -> None:
    """The subprocess and in-process drivers must not drift apart.

    They are two ways to run the same compiler agent; a different default
    on each makes compile cost depend on which driver happened to be
    selected, which is invisible to the user.
    """
    assert CLI_DEFAULT_MODEL == API_DEFAULT_MODEL


def test_max_turns_leaves_headroom_for_bdr_scale_skills() -> None:
    """The turn budget stays >= 60.

    Not an equality check: raising it is harmless, lowering it silently
    truncates a compile into `error_max_turns` after the run has already
    spent most of its money.
    """
    assert DEFAULT_MAX_TURNS >= _MIN_MAX_TURNS, (
        f"DEFAULT_MAX_TURNS is {DEFAULT_MAX_TURNS}. BDR-scale skills need "
        f"~25 tool calls minimum and 40-50 with exploration; the original "
        f"default of 30 caused a hard error_max_turns failure on the first "
        f"real run. Don't reduce it without measuring."
    )
