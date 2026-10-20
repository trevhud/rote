"""Emitted TypeScript must compile against real, typed implementations.

Every TS runtime emits extracted stubs as ``Promise<never>``, and
``never`` converts to anything. So a cast at a reference site that is
wrong for a real output type compiles fine against the stubs, and only
fails the moment a user types the implementation. That is how
``"generate_report": generate_report_result as Record<string, unknown>``
shipped on all three runtimes: ops-report's ``generate_report`` declares
``output: report_markdown str``, and ``string`` does not overlap a
Record (TS2352). The field-access casts had the same hole for an
``interface`` result, which has no implicit index signature.

This suite emits one pipeline covering every output shape the IR
allows, replaces each stub with an implementation typed the way a user
would write it, and runs ``tsc --noEmit``. The stubs are compiled too,
so a fix for real types can't regress the scaffold. The negative half
renames one field ``render`` reads and requires tsc to reject exactly
that: a cast at the field-access site would compile and hand the next
node ``undefined``. On Inngest it is the only half that catches a
restored ``as Record`` field cast, because ``step.run`` returns
``Jsonify<T>``, an anonymous object type that overlaps a Record.

| Node         | Shape                              | Referenced as            |
|--------------|------------------------------------|--------------------------|
| fetch_page   | scalar (``Promise<string>``)       | whole output, exit       |
| list_items   | list of interfaces                 | fanned list, exit        |
| score_item   | fan_out of an interface            | whole output, exit       |
| summarize    | object with declared keys (interface) | field access, exit    |
| render       | scalar                             | exit                     |

Gated behind ``@pytest.mark.slow``: needs Node/npm and the npm registry.
No inference: tsc only.
"""

from __future__ import annotations

import shutil
import subprocess
import textwrap
from pathlib import Path
from typing import Any

import pytest
import yaml

from rote.adapters._common import _to_camel_case
from rote.adapters.cloudflare import CloudflareAdapter
from rote.adapters.dbos_ts import DbosTsAdapter
from rote.adapters.inngest import InngestAdapter
from rote.ir import Pipeline
from tests import _node_trees as node_trees

pytestmark = pytest.mark.slow

_PIPELINE_YAML = """\
name: typed-shapes
version: "0.1.0"
input:
  type: ShapesInput
  required: []
  optional: [title]
nodes:
  - id: fetch_page
    kind: external_call
    description: Fetch a page as text.
    impl: extracted/fetch_page.py:fetch_page
    output: page_text str
  - id: list_items
    kind: external_call
    description: List the items to score.
    impl: extracted/list_items.py:list_items
    output: items list
  - id: score_item
    kind: pure_function
    description: Score one item.
    impl: extracted/score_item.py:score_item
    fan_out: true
    inputs:
      item: list_items.output
    output:
      name: str
      score: int
  - id: summarize
    kind: pure_function
    description: Summarize the scores.
    impl: extracted/summarize.py:summarize
    inputs:
      scores: score_item.output
      page: fetch_page.output
    output:
      names: list
      total: int
  - id: render
    kind: pure_function
    description: Render the report.
    impl: extracted/render.py:render
    inputs:
      names: summarize.output.names
      total: summarize.output.total
      page: fetch_page.output
      title: pipeline.input.title
    output: report str
  - id: extract
    kind: llm_judge
    description: Extract free-form records from the page.
    inputs:
      page: fetch_page.output
    signature_spec:
      input_schema:
        type: object
        properties:
          page: {type: string}
        required: [page]
      # Free-form objects, in an array and nested, plus an untyped field:
      # a Cloudflare step.do result must be Rpc.Serializable, which
      # Record<string, unknown> is not (deal-monitor's shape, from rote bench).
      # The client is openai because an anthropic judge with this schema is
      # refused at emit: structured outputs cannot express free-form objects.
      output_schema:
        type: object
        properties:
          records: {type: array, items: {type: object}}
          meta: {type: object}
          nested:
            type: object
            properties:
              extra: {type: object}
              anything: {}
            required: [extra, anything]
        required: [records, meta, nested]
      prompt: "Extract from {{ page }}"
      client: openai
edges:
  - {from: list_items, to: score_item}
  - {from: score_item, to: summarize}
  - {from: fetch_page, to: summarize}
  - {from: summarize, to: render}
  - {from: fetch_page, to: render}
  - {from: fetch_page, to: extract}
entry_nodes: [fetch_page, list_items]
exit_nodes: [render, summarize, score_item, list_items, fetch_page, extract]
"""

# Typed the way a user fills a stub in: concrete return types, inputs
# narrowed from the `unknown` the stub signature promises. Lists carry
# more than one element so nothing here is a degenerate no-op.
_IMPLEMENTATIONS: dict[str, str] = {
    "fetch_page": """\
        export async function fetchPage(_input: unknown): Promise<string> {
            return "<h1>Quarterly scores</h1>";
        }
        """,
    "list_items": """\
        export interface Item {
            name: string;
            weight: number;
        }

        export async function listItems(_input: unknown): Promise<Item[]> {
            return [
                { name: "alpha", weight: 3 },
                { name: "beta", weight: 5 },
                { name: "gamma", weight: 8 },
            ];
        }
        """,
    "score_item": """\
        import type { Item } from "./list_items";

        export interface Score {
            name: string;
            score: number;
        }

        export async function scoreItem(input: unknown): Promise<Score> {
            const { item } = input as { item: Item };
            return { name: item.name, score: item.weight * 10 };
        }
        """,
    "summarize": """\
        import type { Score } from "./score_item";

        export interface Summary {
            names: string[];
            total: number;
        }

        export async function summarize(input: unknown): Promise<Summary> {
            const { scores } = input as { scores: Score[] };
            return {
                names: scores.map((s) => s.name),
                total: scores.reduce((sum, s) => sum + s.score, 0),
            };
        }
        """,
    "render": """\
        export async function render(input: unknown): Promise<string> {
            const { names, total, title } = input as {
                names: string[];
                total: number;
                title?: string;
            };
            return `# ${title ?? "Report"}\\n${names.join(", ")}: ${total}`;
        }
        """,
}

_ADAPTERS: dict[str, Any] = {
    "cloudflare": CloudflareAdapter,
    "dbos-ts": DbosTsAdapter,
    "inngest": InngestAdapter,
}


def _node_available() -> bool:
    return shutil.which("node") is not None and shutil.which("npm") is not None


def _tsc(cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["npx", "--no-install", "tsc", "--noEmit"],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=180,
    )


@pytest.fixture(scope="module", params=sorted(_ADAPTERS))
def emitted(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> Path:
    if not _node_available():
        pytest.skip("Node / npm not available")
    pipeline = Pipeline.model_validate(yaml.safe_load(_PIPELINE_YAML))
    out = tmp_path_factory.mktemp(f"typed-{request.param}")
    _ADAPTERS[request.param]().emit(pipeline, out)
    node_trees.install(out)
    return out


#: The emitted stubs, captured before any test overwrites them.
_STUBS: dict[Path, dict[str, str]] = {}


@pytest.fixture
def fresh(emitted: Path) -> Path:
    """The emitted dir with its original stubs back in place, so each test
    starts from the scaffold regardless of what an earlier one wrote."""
    extracted = emitted / "src" / "extracted"
    if emitted not in _STUBS:
        _STUBS[emitted] = {n: (extracted / f"{n}.ts").read_text() for n in _IMPLEMENTATIONS}
    for node_id, stub in _STUBS[emitted].items():
        # Replace the real stub, not a file the workflow never imports.
        assert f"export async function {_to_camel_case(node_id)}(" in stub
        assert "Promise<never>" in stub
        (extracted / f"{node_id}.ts").write_text(stub)
    return emitted


def _write_implementations(emitted: Path, overrides: dict[str, str] | None = None) -> None:
    for node_id, source in {**_IMPLEMENTATIONS, **(overrides or {})}.items():
        (emitted / "src" / "extracted" / f"{node_id}.ts").write_text(textwrap.dedent(source))


def test_stubs_and_typed_implementations_compile(fresh: Path) -> None:
    stubs = _tsc(fresh)
    assert stubs.returncode == 0, f"tsc failed on the emitted stubs:\n{stubs.stdout}"

    _write_implementations(fresh)
    typed = _tsc(fresh)
    assert typed.returncode == 0, (
        f"tsc failed once the stubs were replaced by typed implementations:\n{typed.stdout}"
    )


# `render` reads `summarize.output.total`; this implementation names the
# field `sum`. Everything else is the positive fixture, so the field name
# is the only thing that can fail.
_DRIFTED_SUMMARIZE = _IMPLEMENTATIONS["summarize"].replace("total", "sum")


def test_a_field_the_implementation_lacks_fails_tsc(fresh: Path) -> None:
    """The negative half: drift between the IR and the user's type must be
    a compile error, not an ``undefined`` handed to the next node at run
    time. Any cast at the field-access site, through ``unknown`` or not,
    silences exactly this."""
    assert "sum:" in _DRIFTED_SUMMARIZE
    assert "total" not in _DRIFTED_SUMMARIZE
    _write_implementations(fresh, {"summarize": _DRIFTED_SUMMARIZE})
    drifted = _tsc(fresh)
    assert drifted.returncode != 0, "tsc accepted a field the implementation does not declare"
    # Failing for the right reason: the missing field, and nothing else.
    # TS7053 is bracket indexing with an absent key under the emitted
    # tsconfig's `strict` (noImplicitAny).
    errors = [line for line in drifted.stdout.splitlines() if "error TS" in line]
    assert errors, drifted.stdout
    assert all("error TS7053" in e and "'\"total\"'" in e for e in errors), drifted.stdout


# The implement skill says to replace the stub's `unknown` input with a real
# interface. These do that for every node that takes inputs: a fanned
# element, a whole upstream output, field reads, and an optional pipeline
# input. On DBOS-TS the step wrapper passed `Record<string, unknown>`, so
# following the skill failed tsc (TS2345).
_CLOSED_INPUTS: dict[str, str] = {
    "score_item": """\
        import type { Item } from "./list_items";

        export interface Score {
            name: string;
            score: number;
        }

        export async function scoreItem(input: { item: Item }): Promise<Score> {
            return { name: input.item.name, score: input.item.weight * 10 };
        }
        """,
    "summarize": """\
        import type { Score } from "./score_item";

        export interface Summary {
            names: string[];
            total: number;
        }

        export interface SummarizeInput {
            scores: Score[];
            page: string;
        }

        export async function summarize(input: SummarizeInput): Promise<Summary> {
            return {
                names: input.scores.map((s) => s.name),
                total: input.scores.reduce((sum, s) => sum + s.score, 0) + input.page.length,
            };
        }
        """,
    "render": """\
        export interface RenderInput {
            names: string[];
            total: number;
            page: string;
            title?: string;
        }

        export async function render(input: RenderInput): Promise<string> {
            return `# ${input.title ?? "Report"}\\n${input.names.join(", ")}: ${input.total}`;
        }
        """,
}


def test_implementations_with_closed_input_interfaces_compile(fresh: Path) -> None:
    """Following the implement skill (a real interface for the input, not
    ``unknown``) must compile on every TS runtime."""
    _write_implementations(fresh, _CLOSED_INPUTS)
    typed = _tsc(fresh)
    assert typed.returncode == 0, (
        f"tsc failed on implementations with closed input interfaces:\n{typed.stdout}"
    )


def test_an_input_the_ir_does_not_send_fails_tsc(fresh: Path) -> None:
    """The negative half: asserting each value to the impl's declared type
    must not hide drift. An interface that requires a key the IR never
    binds is a compile error on every runtime (TS2741), not ``undefined``
    at run time."""
    extra = _CLOSED_INPUTS["summarize"].replace(
        "    page: string;\n", "    page: string;\n    region: string;\n"
    )
    assert "region: string;" in extra
    _write_implementations(fresh, {**_CLOSED_INPUTS, "summarize": extra})
    drifted = _tsc(fresh)
    assert drifted.returncode != 0, "tsc accepted an input the IR never sends"
    errors = [line for line in drifted.stdout.splitlines() if "error TS" in line]
    # One error, at the summarize call, naming the key.
    assert len(errors) == 1, drifted.stdout
    assert "Property 'region' is missing" in drifted.stdout, drifted.stdout


def test_an_input_the_impl_does_not_declare_fails_tsc(fresh: Path) -> None:
    """And the other direction: the IR binds ``page`` to ``summarize``, so
    an interface without it fails (TS2339 on the declared-type lookup)."""
    missing = (
        _CLOSED_INPUTS["summarize"]
        .replace("    page: string;\n", "")
        .replace(" + input.page.length", "")
    )
    assert "page" not in missing
    _write_implementations(fresh, {**_CLOSED_INPUTS, "summarize": missing})
    drifted = _tsc(fresh)
    assert drifted.returncode != 0, "tsc accepted an input the impl never declares"
    errors = [line for line in drifted.stdout.splitlines() if "error TS" in line]
    assert errors and all('"page"' in e or "'page'" in e for e in errors), drifted.stdout
