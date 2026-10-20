"""Registrable domains, by the Public Suffix List (ICANN and private sections).

The list is bundled, pinned, and never downloaded at run time:
``public_suffix_list.dat.gz`` is the file at publicsuffix/list commit
:data:`COMMIT`, compressed, whose text hashes to :data:`SHA256`
(``tests/test_psl.py`` checks both). Update it by fetching that file at a
newer commit and changing the two constants.

The private section matters here: ``a.github.io`` and ``b.github.io`` are
different owners, and so are two ``herokuapp.com`` apps.

https://publicsuffix.org/list/ (the algorithm is under "Formal Algorithm")
"""

from __future__ import annotations

import gzip
from functools import cache
from importlib import resources

#: publicsuffix/list commit the bundled file was taken from (2026-10-20).
COMMIT = "5dd321ffe02039e2431382a095cf80e27ff5eb94"
SHA256 = "9b4827a9e428a0192a9ea0a6b7804a4cc653d5f39a03456d3b17447b86d78de9"
_FILE = "public_suffix_list.dat.gz"


def list_text() -> str:
    """The bundled list, as published."""
    raw = resources.files("rote.vendor_specs").joinpath(_FILE).read_bytes()
    return gzip.decompress(raw).decode("utf-8")


@cache
def _rules() -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
    """``(rules, wildcards, exceptions)``: ``co.uk``; ``ck`` for ``*.ck``;
    ``www.ck`` for ``!www.ck``. Labels as IDNA ASCII, lowercase."""
    rules: set[str] = set()
    wildcards: set[str] = set()
    exceptions: set[str] = set()
    for line in list_text().splitlines():
        rule = line.strip().split(" ", 1)[0]
        if not rule or rule.startswith("//"):
            continue
        if rule.startswith("!"):
            exceptions.add(_ascii(rule[1:]))
        elif rule.startswith("*."):
            wildcards.add(_ascii(rule[2:]))
        else:
            rules.add(_ascii(rule))
    return frozenset(rules), frozenset(wildcards), frozenset(exceptions)


def _ascii(name: str) -> str:
    try:
        return name.encode("idna").decode("ascii").lower()
    except UnicodeError:
        return name.lower()


def public_suffix(host: str) -> str:
    """The longest public suffix of ``host`` (``co.uk`` for ``bbc.co.uk``)."""
    labels = _ascii(host.rstrip(".")).split(".")
    rules, wildcards, exceptions = _rules()
    for i in range(len(labels)):
        candidate = ".".join(labels[i:])
        if candidate in exceptions:
            # An exception rule: the suffix is the rule without its first label.
            return ".".join(labels[i + 1 :])
        if candidate in rules or ".".join(labels[i + 1 :]) in wildcards:
            return candidate
    return labels[-1]  # the default rule, "*"


def registrable_domain(host: str) -> str:
    """The public suffix plus one label (``slack.com`` for ``api.slack.com``).

    An IP literal, a one-label host, or a host that is itself a public
    suffix (``github.io``) is itself.
    """
    host = host.lower().rstrip(".")
    labels = host.split(".")
    if len(labels) < 2 or ":" in host or all(part.isdigit() for part in labels):
        return host
    suffix = public_suffix(host)
    count = suffix.count(".") + 1
    if count >= len(labels):
        return host
    return ".".join(labels[-(count + 1) :])


__all__ = ["COMMIT", "SHA256", "list_text", "public_suffix", "registrable_domain"]
