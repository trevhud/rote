"""A docs spec is held to the pages it cites, in code.

A docs spec is the agent's own claim about a vendor that publishes no
machine-readable spec, and the agent also writes the code checked against
it. Without an outside anchor the check is circular: in #173's own run the
check bounced a read of ``error``, and the agent added ``error`` to its
spec. So rote fetches each page the spec cites itself (a plain GET, no
credential, no inference) and requires every name the spec declares to
occur in that text:

* the operation's last literal path segment (``conversations.history``),
* every parameter name,
* every key of the request and response examples, nested ones included.

A name found in no cited source is ``docs_spec_unsupported``: an error that
goes back to the agent with the repair loop. ``skill`` as a source means the
skill's own SKILL.md and references. A spec none of whose sources can be
read is ``docs_spec_ungrounded``, a warning, the same as an unreachable
vendor spec: an offline compile is not failed for it.

Matching is on word boundaries, case-sensitive: field names are.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from rote.contracts import ContractFinding
from rote.vendor_specs.docs import DOCS_DIRNAME, DocsSpecFile

PAGE_TTL_SECONDS = 24 * 3600
_FETCH_TIMEOUT_SECONDS = 30
#: The most of one docs page the check reads.
MAX_PAGE_BYTES = 4 * 1024 * 1024
_DROPPED = re.compile(r"<(script|style|noscript)\b.*?</\1\s*>", re.IGNORECASE | re.DOTALL)
_TAG = re.compile(r"<[^>]+>")
_PLACEHOLDER = re.compile(r"\{[^}]*\}")


class PageUnavailable(RuntimeError):
    """A cited page could not be fetched."""


def http_get_text(url: str) -> str:
    """GET ``url`` as text. The one network call in this module."""
    request = urllib.request.Request(url, headers={"User-Agent": "rote-docs-grounding"})
    try:
        with urllib.request.urlopen(request, timeout=_FETCH_TIMEOUT_SECONDS) as response:  # noqa: S310
            raw = response.read(MAX_PAGE_BYTES)
            charset = response.headers.get_content_charset() or "utf-8"
    except (urllib.error.URLError, OSError, ValueError) as e:
        raise PageUnavailable(f"could not fetch {url}: {e}") from e
    return bytes(raw).decode(str(charset), errors="replace")


def page_text(url: str, cache_root: Path, fetch: Callable[[str], str] | None = None) -> str | None:
    """The readable text of ``url`` (cached a day), or ``None`` when it cannot be fetched."""
    path = cache_root / f"{hashlib.sha256(url.encode()).hexdigest()[:24]}.json"
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
        if time.time() - float(cached["fetched_at"]) <= PAGE_TTL_SECONDS:
            return str(cached["text"])
    except (OSError, ValueError, KeyError):
        pass
    try:
        text = to_text((fetch or http_get_text)(url))
    except PageUnavailable:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"url": url, "fetched_at": time.time(), "text": text}), "utf-8")
    return text


def to_text(page: str) -> str:
    """A page's text: scripts and styles dropped, tags removed, entities decoded."""
    if "<" not in page:
        return page
    return html.unescape(_TAG.sub(" ", _DROPPED.sub(" ", page)))


def declared_names(op: Any) -> Iterator[tuple[str, str]]:
    """``(what, name)`` for every name a docs operation declares."""
    path = op.url_template.split("?", 1)[0].split("://", 1)[-1].split("/", 1)
    literals = [s for s in (path[1] if len(path) > 1 else "").split("/") if s]
    literals = [s for s in literals if not _PLACEHOLDER.fullmatch(s)]
    if literals:
        yield "path segment", _PLACEHOLDER.sub("", literals[-1]).strip(":") or literals[-1]
    for name in op.params:
        yield "parameter", name
    for what, example in (
        ("request field", op.request_example),
        ("response field", op.response_example),
    ):
        for key in _keys(example):
            yield what, key


def _keys(value: Any) -> Iterator[str]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _keys(item)


def occurs(name: str, text: str) -> bool:
    return re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", text) is not None


def skill_text(skill_dir: Path | None) -> str | None:
    if skill_dir is None:
        return None
    parts = [skill_dir / "SKILL.md", *sorted((skill_dir / "references").glob("*.md"))]
    texts = [p.read_text(encoding="utf-8", errors="replace") for p in parts if p.is_file()]
    return "\n".join(texts) if texts else None


def docs_grounding_findings(
    root: Path,
    skill_dir: Path | None,
    cache_root: Path,
    fetch: Callable[[str], str] | None = None,
) -> list[ContractFinding]:
    """Every name a docs spec declares that none of its cited sources contains."""
    findings: list[ContractFinding] = []
    for path in sorted((root / DOCS_DIRNAME).glob("*.json")):
        rel = path.relative_to(root).as_posix()
        try:
            raw = DocsSpecFile.model_validate_json(path.read_text(encoding="utf-8"))
        except ValueError:
            continue  # docs_spec_invalid, from the request check
        texts: list[str] = []
        unread: list[str] = []
        for source in raw.sources:
            text = (
                skill_text(skill_dir) if source == "skill" else page_text(source, cache_root, fetch)
            )
            if text is None:
                unread.append(source)
            else:
                texts.append(text)
        if not texts:
            findings.append(
                ContractFinding(
                    node_id="(deliverable)",
                    kind="docs_spec_ungrounded",
                    severity="warning",
                    message=(
                        f"{rel}: none of its sources could be read ({', '.join(unread)}), so "
                        f"its names were not checked against the docs"
                    ),
                    path=rel,
                )
            )
            continue
        corpus = "\n".join(texts)
        for op in raw.operations:
            missing = sorted(
                {(what, name) for what, name in declared_names(op) if not occurs(name, corpus)}
            )
            for what, name in missing:
                findings.append(
                    ContractFinding(
                        node_id="(deliverable)",
                        kind="docs_spec_unsupported",
                        severity="error",
                        message=(
                            f"{rel}: {op.id} declares the {what} {name!r}, which none of its "
                            f"cited sources contains. Copy names from the page as it states "
                            f"them, cite the page that does, or remove it; a spec may not "
                            f"say more than its docs"
                        ),
                        path=rel,
                    )
                )
    return findings


__all__ = [
    "MAX_PAGE_BYTES",
    "PAGE_TTL_SECONDS",
    "PageUnavailable",
    "declared_names",
    "docs_grounding_findings",
    "http_get_text",
    "occurs",
    "page_text",
    "to_text",
]
