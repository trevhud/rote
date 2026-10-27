"""Which environment variables a node may declare it reads (``Node.env``).

``rote skill run`` starts its driver with an allowlisted environment
(:func:`rote.skill_target.launch.child_env`), so a module's own variable
reaches it only when the IR declares it. The sets here are that allowlist's
own, one copy for :mod:`rote.skill_target.launch`, :mod:`rote.skill_target.trust`
and the IR's validator, so a name the allowlist keeps out can never be let
back in by a declaration.

Three classes of name:

- **passed**: every run gets it (:data:`BASE_VARS`, ``LC_*``), so a module
  reads it without declaring it, and declaring it says nothing.
- **reserved**: the allowlist exists to keep it out or to set it itself:
  rote's own ``ROTE_*`` variables (state paths, endpoint overrides, write
  consent, inference knobs), the vendor credentials and endpoints a judge
  pays with, the proxy and CA variables (bound at install, never taken from
  the caller), ``XDG_*``, and what changes how a process loads code
  (``PYTHON*``, ``LD_*``, ``DYLD_*``). No node may declare one.
- **declarable**: everything else, such as ``EXPENSE_AUDIT_REPORT_DIR``.

Stdlib only, and nothing from rote: the IR imports it.
"""

from __future__ import annotations

import re

#: Passed through when set: what any process needs to start and find its tools.
BASE_VARS = frozenset(
    {"PATH", "HOME", "USER", "LOGNAME", "LANG", "LANGUAGE", "TZ", "TMPDIR", "SYSTEMROOT"}
)
#: The vendor keys a ``--judges helper`` skill's judges pay with.
HELPER_KEYS = frozenset(
    {
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "OPENAI_API_KEY",
        "ROTE_INFERENCE",
        "ROTE_INFERENCE_TIMEOUT",
        "ROTE_JUDGE_EFFORT",
    }
)
#: Per-judge knobs a helper skill honors, ``<prefix><JUDGE ID>``.
HELPER_KNOBS = ("ROTE_INFERENCE_", "ROTE_MODEL_", "ROTE_EFFORT_", "ROTE_TEMPERATURE_")
#: Where a vendor SDK sends a judge's payload and key; a skill run drops them.
ENDPOINT_VARS = frozenset({"ANTHROPIC_BASE_URL", "OPENAI_BASE_URL"})
#: Proxy variables a run may use, recorded at install and passed only as recorded.
PROXY_VARS = (
    "HTTPS_PROXY",
    "https_proxy",
    "HTTP_PROXY",
    "http_proxy",
    "ALL_PROXY",
    "all_proxy",
    "NO_PROXY",
    "no_proxy",
)
#: CA bundle variables, recorded by path and the bundle's hash: a proxy plus a
#: CA the attacker chose reads every bearer token.
CA_VARS = ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE")

#: Prefixes no declaration may use. ``ROTE_`` covers every rote variable,
#: the state-path overrides and :data:`HELPER_KNOBS` included.
RESERVED_PREFIXES = ("ROTE_", "XDG_", "PYTHON", "LD_", "DYLD_")

#: The shape of a name a node may declare: what a POSIX shell can export.
ENV_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def always_passed(name: str) -> bool:
    """Whether every skill run passes ``name`` (when set), declared or not."""
    return name in BASE_VARS or name.startswith("LC_")


def reserved_reason(name: str) -> str | None:
    """Why no node may declare ``name``, or ``None`` when it may."""
    if always_passed(name):
        return f"{name} reaches every run already, so declaring it says nothing"
    if name.startswith("ROTE_"):
        return f"{name} is rote's own (state paths, endpoints, consent, inference knobs)"
    if name in HELPER_KEYS or name in ENDPOINT_VARS:
        return f"{name} decides who a judge's call bills and where it goes"
    if name in PROXY_VARS or name in CA_VARS:
        return f"{name} is bound at install, never taken from the caller"
    if name.startswith(RESERVED_PREFIXES):
        return f"{name} changes where state lives or how a process loads code"
    return None
