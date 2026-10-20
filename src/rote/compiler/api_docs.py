"""``--backend api``: the vendor docs a compile relied on, and the proof it read them.

With ``--backend api`` every ``external_call`` node runs its ``impl:``
function in production, and an ``mcp:`` binding on it is never used. So
the agent writes direct vendor-SDK calls, and a call written from
training-data memory is the failure this module exists to expose: SDK and
endpoint shapes drift faster than any model's knowledge, and a
plausible-but-wrong call fails at run time, not at review.

Three parts, one per party:

* :func:`brief` tells the agent the backend and what it owes: research
  the current docs with its web tools and cite each page it relied on as a
  ``# docs: <url>`` comment on the function.
* :func:`docs_findings` holds the deliverable to it. An implemented vendor
  call that cites no page is sent back to the agent.
* :class:`WebResearch` records, from the drivers' ``tool`` events, every
  page the agent actually fetched and every search it ran. The compile
  report lists each cited URL beside whether it was fetched in this
  compile (:func:`cited_docs`), so a reviewer checks the API shape against
  the page, and a citation the agent never opened stands out.
"""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from rote.compiler.events import CompilationEvent, EventCallback, emit_safely
from rote.contracts import ContractFinding, NodeImplementation
from rote.ir import NodeKind, Pipeline

#: ``# docs: https://...``: one per documentation page a function relies on.
#: ``# docs: skill`` instead says the call's contract is the skill's own text
#: (an internal API with no public reference), which its ``# skill:``
#: citations quote.
_DOCS_RE = re.compile(r"#[ \t]*docs:[ \t]*(?P<url>https?://[^\s'\"<>]+|skill\b)")

#: The ``# docs:`` value for "the skill documents this API".
SKILL_DOCS = "skill"

#: A ``# skill: "..."`` quote of the source skill.
_SKILL_QUOTE_RE = re.compile(r'#[ \t]*skill:[ \t]*"')

#: Trailing punctuation a URL written in prose often picks up.
_URL_TAIL = ".,;:)]}"


def brief() -> str:
    """The instruction block every ``--backend api`` compile adds to the prompt.

    Tool names differ per driver (Claude Code's ``WebSearch`` /
    ``WebFetch``, the Messages API's ``web_search`` / ``web_fetch``, Codex's
    web search), so the brief names them by what they do.
    """
    return (
        "BACKEND: this compile targets `--backend api`. In production every "
        "external_call node runs its `impl:` function, and an `mcp:` binding on "
        "the node is never used. Each external_call module is therefore a direct "
        "call to the vendor's API, written under references/implementation.md "
        "Regime 2. You have web search and web fetch tools for this compile:\n"
        "1. Before writing any vendor call, look it up: search for the vendor's "
        "current API reference, then FETCH the reference page for the exact "
        "endpoint or SDK method (a search snippet is not the docs). Run the "
        "lookups for every external_call node in one turn of parallel tool "
        "calls. Your training data is not a source for an endpoint, a parameter, "
        "or a response field.\n"
        "2. Cite every page you relied on with a `# docs: <url>` comment directly "
        "above the function, one line per page. rote lists each cited URL in "
        "compile-report.md beside whether you fetched it during this compile, "
        "and sends back an implemented vendor call that cites none.\n"
        "3. The function returns the node's declared `output` contract, built "
        "from the documented response: downstream nodes consume that shape, not "
        "the vendor's raw payload and not the MCP tool's.\n"
        "4. If the docs do not let you confirm a request or response shape, "
        "write a stub whose NotImplementedError message names the page and what "
        "it left unconfirmed, rather than a call from memory. A page you cite "
        "but never fetched in this compile is sent back to you.\n"
        "5. Where the vendor publishes a machine-readable spec (Google APIs: the "
        "Discovery document), rote checks every request against it in code: the "
        "method and path, each parameter's name, type and enum, the required "
        "parameters, and every response field the code reads. Write requests a "
        "static reader can follow: literal URLs or module constants, and params "
        "as a dict literal. Your tests run with the vendor libraries you import "
        "installed: stub their transport (httpx.MockTransport, googleapiclient's "
        "HttpMock), never the library itself. Where the vendor publishes no "
        "machine-readable spec, rote asks you to write one from the pages you "
        'fetched (implementation.md, "Specs written from the docs").\n'
        "6. An internal or undocumented API: if the skill or its references state "
        "the endpoint and shapes, implement from that text, quote it in "
        "`# skill:` citations, and write `# docs: skill` above the function. If "
        "nothing documents it, write a stub naming what is missing. Never guess "
        "an endpoint."
    )


def api_view(pipeline: Pipeline) -> Pipeline:
    """``pipeline`` as the api backend runs it: no ``mcp:`` binding is called.

    Every bound external_call runs its ``impl:`` instead (an
    ``agent_loop``'s tools are untouched: they are MCP on every backend).
    Checks about a binding's tool, its arguments, its recorded result and
    the server's auth, run on this view, have nothing to report.
    """
    nodes = [
        n.model_copy(update={"mcp": None}) if n.kind is NodeKind.EXTERNAL_CALL else n
        for n in pipeline.nodes
    ]
    return pipeline.model_copy(update={"nodes": nodes})


def normalize_url(url: str) -> str:
    """One spelling per page: no fragment, no trailing slash, lowercase host.

    ``http`` and ``https`` are the same page for this purpose, and a
    ``#section`` anchor points into the page that was fetched.
    """
    url = url.strip().rstrip(_URL_TAIL)
    if url == SKILL_DOCS:
        return url
    parts = urlsplit(url)
    path = parts.path.rstrip("/")
    return urlunsplit(("https", parts.netloc.lower(), path, parts.query, ""))


@dataclass(frozen=True)
class DocCitation:
    """One ``# docs:`` URL in an extracted module."""

    node_id: str
    url: str
    path: str
    line: int
    #: The line ranges that speak for the node's function.
    spans: tuple[tuple[int, int], ...] = ()


def module_docs(source: str) -> list[tuple[int, str]]:
    """``(line, url)`` for every ``# docs:`` comment in ``source``."""
    found: list[tuple[int, str]] = []
    for number, text in enumerate(source.splitlines(), start=1):
        for match in _DOCS_RE.finditer(text):
            found.append((number, match.group("url").rstrip(_URL_TAIL)))
    return found


def cited_docs(pipeline: Pipeline, root: Path) -> list[DocCitation]:
    """Every ``# docs:`` page an ``external_call`` node's function relies on, once each.

    A citation belongs to a node's function when it sits in the function,
    in the lines between the previous top-level statement and the ``def``
    (the comment directly above it), or in the module's header before the
    first top-level ``def`` or ``class``, which speaks for every function
    in the module.
    """
    out: list[DocCitation] = []
    for node in pipeline.nodes:
        if node.kind is not NodeKind.EXTERNAL_CALL or not node.impl:
            continue
        rel, _, symbol = node.impl.partition(":")
        try:
            source = (root / rel).read_text(encoding="utf-8")
            tree = ast.parse(source)
        except (OSError, UnicodeDecodeError, SyntaxError):
            continue
        spans = _function_spans(tree, symbol)
        seen: set[str] = set()
        for line, url in module_docs(source):
            key = normalize_url(url)
            if key in seen or not any(lo <= line <= hi for lo, hi in spans):
                continue
            seen.add(key)
            out.append(DocCitation(node.id, url, rel, line, tuple(spans)))
    return out


def _function_spans(tree: ast.Module, symbol: str) -> list[tuple[int, int]]:
    """The line ranges whose ``# docs:`` comments speak for ``symbol``.

    The header runs to the end of the last statement before the first
    top-level ``def`` or ``class``; a comment after that and above a
    ``def`` is that function's.
    """
    defs = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    header_end = 0
    for stmt in tree.body:
        if isinstance(stmt, defs):
            break
        header_end = stmt.end_lineno or header_end
    else:
        header_end = 10**9  # no def at all: the whole module is header
    spans = [(1, header_end)]
    previous_end = 0
    for stmt in tree.body:
        if (
            isinstance(stmt, ast.FunctionDef | ast.AsyncFunctionDef)
            and stmt.name == symbol
            and stmt.end_lineno is not None
        ):
            spans.append((previous_end + 1, stmt.end_lineno))
        previous_end = stmt.end_lineno or previous_end
    return spans


def docs_findings(
    pipeline: Pipeline,
    root: Path,
    implementations: list[NodeImplementation],
    research: WebResearch | None = None,
) -> list[ContractFinding]:
    """What the vendor calls' docs citations fail to show, under ``--backend api``.

    * An implemented vendor call that cites no docs page. A stub is not held
      to it: its NotImplementedError message is where the agent says which
      page left the shape unconfirmed. A citation counts when
      :func:`cited_docs` attributes it to the node's function.
    * A cited page the agent never fetched in this compile, when the driver
      reports its fetches (``research.reported``): the call may come from
      memory, with a real URL attached. ``# docs: skill`` is exempt.
    * A ``# docs: skill`` with no ``# skill: "..."`` quote speaking for the
      same function: "the skill documents this API" with nothing quoted
      is a call from memory under another name.
    """
    cites = cited_docs(pipeline, root)
    cited = {c.node_id for c in cites}
    kinds = {n.id: n.kind for n in pipeline.nodes}
    findings: list[ContractFinding] = []
    for impl in implementations:
        if kinds.get(impl.node_id) is not NodeKind.EXTERNAL_CALL:
            continue
        if impl.status != "implemented" or impl.node_id in cited:
            continue
        rel, _, symbol = impl.impl.partition(":")
        findings.append(
            ContractFinding(
                node_id=impl.node_id,
                kind="uncited_api_call",
                severity="error",
                message=(
                    f"{symbol}() calls a vendor API but its module cites no docs page. "
                    f"Fetch the vendor's current reference for this call and add a "
                    f"`# docs: <url>` comment above the function for each page you "
                    f"relied on (implementation.md, Regime 2)"
                ),
                path=rel,
            )
        )
    for c in cites:
        if c.url != SKILL_DOCS or _quotes_skill(root, c):
            continue
        findings.append(
            ContractFinding(
                node_id=c.node_id,
                kind="unquoted_skill_docs",
                severity="error",
                message=(
                    f"{c.path}:{c.line} says the skill documents this API (`# docs: "
                    f"skill`), but the function quotes none of it. Quote the skill text "
                    f'that states the endpoint and shapes in `# skill: "..."` citations, '
                    f"or cite the vendor's docs page (implementation.md, Regime 2)"
                ),
                path=c.path,
            )
        )
    if research is not None and research.reported:
        for c in cites:
            if c.url == SKILL_DOCS or research.was_fetched(c.url):
                continue
            findings.append(
                ContractFinding(
                    node_id=c.node_id,
                    kind="unfetched_docs_page",
                    severity="error",
                    message=(
                        f"{c.path}:{c.line} cites {c.url}, which this compile never "
                        f"fetched. Fetch it and check the call against it, or cite the "
                        f"page you did read (implementation.md, Regime 2)"
                    ),
                    path=c.path,
                )
            )
    return findings


def _quotes_skill(root: Path, citation: DocCitation) -> bool:
    """Whether a ``# skill: "..."`` quote sits in the lines that speak for
    the citation's function (the same spans as its ``# docs:`` comments)."""
    source = (root / citation.path).read_text(encoding="utf-8")
    return any(
        _SKILL_QUOTE_RE.search(text) and any(lo <= n <= hi for lo, hi in citation.spans)
        for n, text in enumerate(source.splitlines(), start=1)
    )


#: Tool names the drivers report a page fetch / a web search under.
FETCH_TOOLS = frozenset({"WebFetch", "web_fetch"})
SEARCH_TOOLS = frozenset({"WebSearch", "web_search"})


#: rote's record, beside the IR, of the pages a compile fetched.
RESEARCH_FILENAME = "web-research.json"


def write_research(compiled_dir: Path, research: WebResearch) -> Path:
    path = compiled_dir / RESEARCH_FILENAME
    path.write_text(
        json.dumps(
            {"schema_version": 1, "fetched": research.fetched, "searches": research.searches},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def read_fetched(compiled_dir: Path) -> set[str] | None:
    """The normalized URLs the compile fetched, or ``None`` when it recorded none."""
    path = compiled_dir / RESEARCH_FILENAME
    if not path.is_file():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {normalize_url(u) for u in raw.get("fetched") or []}


@dataclass
class WebResearch:
    """What the agent looked up during one compile, from the drivers' ``tool`` events.

    ``reported`` is False when the driver emitted no web tool event at all,
    which for a driver that cannot report them (see each driver) is not the
    same as an agent that did no research.
    """

    fetched: list[str] = field(default_factory=list)
    searches: list[str] = field(default_factory=list)

    @property
    def reported(self) -> bool:
        return bool(self.fetched or self.searches)

    def observe(self, event: CompilationEvent) -> None:
        if event.type != "tool":
            return
        if event.tool_name in FETCH_TOOLS and event.url and event.url not in self.fetched:
            self.fetched.append(event.url)
        elif event.tool_name in SEARCH_TOOLS and event.query and event.query not in self.searches:
            self.searches.append(event.query)

    def tee(self, sink: EventCallback | None) -> Callable[[CompilationEvent], None]:
        """An event callback that records, then forwards to ``sink``."""

        def _both(event: CompilationEvent) -> None:
            self.observe(event)
            emit_safely(sink, event)

        return _both

    def was_fetched(self, url: str) -> bool:
        return normalize_url(url) in {normalize_url(u) for u in self.fetched}


def report_section(pipeline: Pipeline, root: Path, research: WebResearch | None) -> list[str]:
    """The compile report's "Vendor API docs" section for an ``--backend api`` compile."""
    cites = cited_docs(pipeline, root)
    lines = [
        "This pipeline was compiled for `--backend api`: each `external_call` runs "
        "its own vendor-API code, not an MCP tool. Check each call against the "
        "page it cites.",
        "",
    ]
    if research is None or not research.reported:
        lines += [
            "The compile reported no web fetches, so whether the agent read these "
            "pages in this compile is unknown.",
            "",
        ]
    if cites:
        lines += ["| Node | Docs page | Where | Fetched in this compile |", "|---|---|---|---|"]
        for c in cites:
            if c.url == SKILL_DOCS:
                page, fetched = "the skill's own text (internal API)", "n/a"
            elif research is None or not research.reported:
                page, fetched = c.url, "unknown"
            else:
                page = c.url
                fetched = "yes" if research.was_fetched(c.url) else "**no**: check it first"
            lines.append(
                f"| `{c.node_id}` | {_cell(page)} | `{_cell(c.path)}:{c.line}` | {fetched} |"
            )
    else:
        lines.append("No `external_call` module cites a docs page.")
    if research is not None and research.reported:
        cited = {normalize_url(c.url) for c in cites}
        uncited = [u for u in research.fetched if normalize_url(u) not in cited]
        lines += [
            "",
            f"The agent ran {len(research.searches)} web search(es) and fetched "
            f"{len(research.fetched)} page(s).",
        ]
        if uncited:
            lines += ["", "Fetched but not cited:", "", *(f"- {u}" for u in uncited)]
    return lines


def _cell(text: str) -> str:
    """One markdown table cell: a ``|`` in a URL would end it."""
    return text.replace("|", "\\|")


__all__ = [
    "FETCH_TOOLS",
    "SEARCH_TOOLS",
    "SKILL_DOCS",
    "DocCitation",
    "WebResearch",
    "api_view",
    "brief",
    "cited_docs",
    "docs_findings",
    "module_docs",
    "normalize_url",
    "report_section",
]
