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
so a fix for real types can't regress the scaffold.

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

import os
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
edges:
  - {from: list_items, to: score_item}
  - {from: score_item, to: summarize}
  - {from: fetch_page, to: summarize}
  - {from: summarize, to: render}
  - {from: fetch_page, to: render}
entry_nodes: [fetch_page, list_items]
exit_nodes: [render, summarize, score_item, list_items, fetch_page]
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
    proc = subprocess.run(
        ["npm", "install", "--no-audit", "--no-fund"],
        cwd=out,
        capture_output=True,
        text=True,
        timeout=300,
        env={**os.environ, "npm_config_progress": "false"},
    )
    if proc.returncode != 0:
        pytest.fail(f"npm install failed in {out}:\n{proc.stdout}\n{proc.stderr}")
    return out


def test_stubs_and_typed_implementations_compile(emitted: Path) -> None:
    stubs = _tsc(emitted)
    assert stubs.returncode == 0, f"tsc failed on the emitted stubs:\n{stubs.stdout}"

    for node_id, source in _IMPLEMENTATIONS.items():
        path = emitted / "src" / "extracted" / f"{node_id}.ts"
        # Replace the real stub, not a file the workflow never imports.
        stub = path.read_text()
        assert f"export async function {_to_camel_case(node_id)}(" in stub
        assert "Promise<never>" in stub
        path.write_text(textwrap.dedent(source))

    typed = _tsc(emitted)
    assert typed.returncode == 0, (
        f"tsc failed once the stubs were replaced by typed implementations:\n{typed.stdout}"
    )
