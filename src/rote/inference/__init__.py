"""Judge inference providers — who serves an emitted judge, and who pays.

The implementation lives in :mod:`rote.inference._runtime_helper`,
which is emitted **verbatim** into generated Python apps as
``signatures/_rote_inference.py`` (the same pattern
:mod:`rote.mcp._runtime_helper` uses for ``extracted/_rote_mcp.py``).
That module is stdlib-only and imports nothing from rote, which is what
makes the copy a copy rather than a fork. This package re-exports the
pieces the CLI itself needs — never a second implementation.

Design record: ``docs/inference-runtime.md``.
"""

from __future__ import annotations

from rote.inference._runtime_helper import (
    PROVIDERS,
    build_subscription_env,
    provider_availability,
    select_provider,
)

#: Providers that cannot exist in a deployed image. A Claude
#: subscription credential is personal, and Cloudflare Workers has no
#: subprocess at all — so choosing this for a deploy target is refused
#: at emit time rather than discovered at 3am in production.
LOCAL_ONLY_PROVIDERS: frozenset[str] = frozenset({"claude-cli"})

__all__ = [
    "LOCAL_ONLY_PROVIDERS",
    "PROVIDERS",
    "build_subscription_env",
    "provider_availability",
    "select_provider",
]
