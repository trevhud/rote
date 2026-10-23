"""Turning one client's server entry into rote's shape, the same way for every client.

Four rules, each a function so every source applies it identically:

- :func:`server_identifier`: rote names are identifiers (the IR's
  ``mcp.server`` rule), so ``acme-crm-prod`` becomes ``acme_crm_prod``
  and the candidate records the original.
- :func:`rewrite_placeholders`: each client's variable syntax becomes
  rote's one form, ``${VAR}``. Nothing is ever resolved: the value of the
  variable stays in the user's environment, where the client read it. A
  default (``${VAR:-x}``) is dropped and reported, since it may be the
  literal secret.
- :func:`header_template`: a header is copied only when its value is
  references plus at most an auth scheme word. Anything else is a literal,
  maybe a secret, and is withheld with the command that sets it by hand.
- :func:`url_problem`: a URL is copied only when it is a plain http(s)
  endpoint: no credentials in it, no query string (where API keys travel),
  no placeholder (rote does not expand variables in URLs).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlsplit

from rote.clients.base import HeaderTemplate

_NOT_IDENTIFIER = re.compile(r"[^A-Za-z0-9_]")
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def server_identifier(name: str) -> str:
    """``name`` as a rote server name: every other character becomes ``_``,
    and a leading digit gets a ``_`` in front."""
    mapped = _NOT_IDENTIFIER.sub("_", name.strip())
    if not mapped or not mapped.strip("_"):
        raise ValueError(f"server name {name!r} has no letters or digits to keep")
    if mapped[0].isdigit():
        mapped = f"_{mapped}"
    assert _IDENTIFIER.fullmatch(mapped)
    return mapped


@dataclass(frozen=True)
class PlaceholderSyntax:
    """How one client writes an environment reference.

    ``pattern`` must define the groups ``var`` (the variable name) and
    optionally ``default``. ``other`` matches the client's references that
    are not environment variables (VS Code's ``${input:id}``, a plugin's
    ``${CLAUDE_PLUGIN_ROOT}``), which rote cannot carry."""

    client: str
    pattern: re.Pattern[str]
    #: Variables the client fills in itself; they are not in the user's
    #: environment, so a rote run would never see them.
    internal: frozenset[str] = frozenset()
    other: re.Pattern[str] | None = None


#: Claude Code: ``${VAR}`` and ``${VAR:-default}`` in url, headers, command,
#: args and env (code.claude.com/docs/en/mcp, "Environment variable
#: expansion"). Plugins add ``${CLAUDE_PLUGIN_ROOT}``, ``${CLAUDE_PLUGIN_DATA}``
#: and ``${CLAUDE_PROJECT_DIR}`` (code.claude.com/docs/en/plugins-reference).
CLAUDE_CODE_SYNTAX = PlaceholderSyntax(
    client="claude-code",
    pattern=re.compile(r"\$\{(?P<var>[A-Za-z_][A-Za-z0-9_]*)(?::-(?P<default>[^}]*))?\}"),
    internal=frozenset({"CLAUDE_PLUGIN_ROOT", "CLAUDE_PLUGIN_DATA", "CLAUDE_PROJECT_DIR"}),
)

#: rote's one form, what every syntax is rewritten to.
ROTE_REFERENCE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


@dataclass(frozen=True)
class Rewritten:
    template: str
    env_vars: tuple[str, ...]
    #: Variables whose ``:-default`` was dropped.
    dropped_defaults: tuple[str, ...]
    #: References rote cannot carry (client-internal variables, inputs).
    unsupported: tuple[str, ...]


def rewrite_placeholders(value: str, syntax: PlaceholderSyntax) -> Rewritten:
    env_vars: list[str] = []
    dropped: list[str] = []
    unsupported: list[str] = []

    def replace(match: re.Match[str]) -> str:
        var = match.group("var")
        if var in syntax.internal:
            unsupported.append(match.group(0))
            return match.group(0)
        if match.groupdict().get("default") is not None:
            dropped.append(var)
        if var not in env_vars:
            env_vars.append(var)
        return f"${{{var}}}"

    template = syntax.pattern.sub(replace, value)
    if syntax.other is not None:
        unsupported.extend(m.group(0) for m in syntax.other.finditer(value))
    return Rewritten(template, tuple(env_vars), tuple(dropped), tuple(unsupported))


#: What may stand beside the references in a copied header value.
_SCHEME_WORDS = frozenset({"", "bearer", "basic", "token"})


@dataclass(frozen=True)
class HeaderVerdict:
    header: HeaderTemplate | None
    #: Why it was not copied, when it was not.
    withheld: str | None = None
    #: Notes on a copied header (a dropped default).
    notes: tuple[str, ...] = ()


def header_template(server: str, name: str, value: str, syntax: PlaceholderSyntax) -> HeaderVerdict:
    """Copy ``name: value`` as a template, or say why not. The value itself
    never appears in a reason."""
    hint = f"set it yourself: rote mcp add {server} <url> --header '{name}: <value>'"
    rewritten = rewrite_placeholders(value, syntax)
    if rewritten.unsupported:
        return HeaderVerdict(
            None,
            f"header {name}: uses {', '.join(rewritten.unsupported)}, which only the client "
            f"fills in; {hint}",
        )
    residual = ROTE_REFERENCE.sub("", rewritten.template).strip().lower()
    if not rewritten.env_vars or residual not in _SCHEME_WORDS:
        return HeaderVerdict(None, f"header {name}: a literal value, not copied; {hint}")
    notes = tuple(
        f"header {name}: the default for ${{{var}}} was not copied (it may be the secret itself); "
        f"set {var} in the environment rote runs in"
        for var in rewritten.dropped_defaults
    )
    return HeaderVerdict(HeaderTemplate(name, rewritten.template, rewritten.env_vars), notes=notes)


def url_problem(url: str, syntax: PlaceholderSyntax) -> str | None:
    """Why ``url`` cannot be copied, or None. The reason never quotes the URL."""
    rewritten = rewrite_placeholders(url, syntax)
    if rewritten.env_vars or rewritten.unsupported:
        names = [*rewritten.env_vars, *rewritten.unsupported]
        return (
            f"the URL depends on {', '.join(names)}, and rote does not expand variables in a "
            "URL; add it with rote mcp add <name> <url>"
        )
    try:
        parts = urlsplit(url)
        _ = parts.port  # raises on a malformed port
    except ValueError:
        return "the URL does not parse"
    if parts.scheme not in ("http", "https"):
        return f"the URL scheme is {parts.scheme or 'missing'}; rote connects over http(s) only"
    if not parts.hostname:
        return "the URL has no host"
    if parts.username is not None or parts.password is not None:
        return "the URL carries credentials; rote does not copy them"
    if parts.query:
        return (
            "the URL has a query string, which can carry an API key, so it is not copied; "
            "add it with rote mcp add <name> <url>"
        )
    return None


def is_work_host(host: str | None, work_domains: tuple[str, ...]) -> bool:
    """Whether ``host`` is, or is under, one of the user's work domains."""
    if host is None:
        return False
    hostname = host.split(":", 1)[0].lower().rstrip(".")
    for domain in work_domains:
        d = domain.lower().strip().lstrip(".").rstrip(".")
        if d and (hostname == d or hostname.endswith(f".{d}")):
            return True
    return False
