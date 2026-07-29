"""The agent_loop contract shared by the three TypeScript runtimes.

Each adapter has its own test module for its own emission shape; this
one covers what all three must agree on, because an agent loop that
behaved differently per runtime would be three implementations of the
loop rather than one.

The pipeline under test is deliberately the awkward case: an agent loop
whose tools are bare names with NO node carrying an ``mcp:`` binding.
That is BDR's shape, and it is where server resolution has nothing to
work from — see :func:`rote.adapters._ts_common.agent_tool_servers`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from rote.adapters._ts_common import ROTE_INFERENCE_HELPER_TS
from rote.adapters.cloudflare import CloudflareAdapter
from rote.adapters.dbos_ts import DbosTsAdapter
from rote.adapters.inngest import InngestAdapter
from rote.ir import Pipeline

# ───────── Fixtures ─────────


def _pipeline(*, with_binding: bool = False) -> Pipeline:
    """A one-loop pipeline; optionally with an MCP-bound sibling node."""
    nodes: list[dict[str, Any]] = [
        {
            "id": "research_loop",
            "kind": "agent_loop",
            "description": "Research the topic until the brief is covered.",
            "tools": ["web_search", "fetch_page"],
            "termination": {"condition": "brief covered", "max_iterations": 4},
            "inputs": {"topic": "pipeline.input.topic"},
        }
    ]
    edges = []
    entry = ["research_loop"]
    if with_binding:
        nodes.insert(
            0,
            {
                "id": "lookup_topic",
                "kind": "external_call",
                "description": "Resolve the topic to a canonical id.",
                "mcp": {
                    "server": "research_tools",
                    "tool": "resolve_topic",
                    "url": "https://mcp.example.com/sse",
                },
                "output": {"topic_id": "string"},
                "inputs": {"topic": "pipeline.input.topic"},
            },
        )
        edges = [{"from": "lookup_topic", "to": "research_loop"}]
        entry = ["lookup_topic"]
    return Pipeline.model_validate(
        {
            "name": "loop-only",
            "version": "0.1.0",
            "source_skill": "tests/fixtures/loop-only",
            "description": "One agent loop.",
            "input": {"type": "Brief", "required": ["topic"], "optional": []},
            "nodes": nodes,
            "edges": edges,
            "entry_nodes": entry,
            "exit_nodes": ["research_loop"],
        }
    )


ADAPTERS = {
    "cloudflare": CloudflareAdapter,
    "dbos-ts": DbosTsAdapter,
    "inngest": InngestAdapter,
}


def _emit(runtime: str, out: Path, *, with_binding: bool = False) -> None:
    ADAPTERS[runtime]().emit(_pipeline(with_binding=with_binding), out)


# ───────── What every TS runtime must agree on ─────────


@pytest.mark.parametrize("runtime", sorted(ADAPTERS))
def test_agent_loop_is_never_emitted_as_a_stub(runtime: str, tmp_path: Path) -> None:
    """The whole point: no runtime hands the agentic step back to the user."""
    _emit(runtime, tmp_path)
    src = (tmp_path / "src" / "extracted" / "research_loop.ts").read_text(encoding="utf-8")
    assert "runAgentLoop({" in src
    assert "implement me" not in src


@pytest.mark.parametrize("runtime", sorted(ADAPTERS))
def test_declared_tools_and_bound_reach_the_loop(runtime: str, tmp_path: Path) -> None:
    """The IR's allowlist IS the boundary — it is emitted verbatim."""
    _emit(runtime, tmp_path)
    src = (tmp_path / "src" / "extracted" / "research_loop.ts").read_text(encoding="utf-8")
    assert 'const TOOLS: string[] = ["web_search", "fetch_page"];' in src
    assert "bindAgentTools(" in src


@pytest.mark.parametrize("runtime", sorted(ADAPTERS))
def test_loop_is_bounded_by_the_ir(runtime: str, tmp_path: Path) -> None:
    """An unbounded loop in a durable workflow burns budget silently."""
    _emit(runtime, tmp_path)
    src = (tmp_path / "src" / "extracted" / "research_loop.ts").read_text(encoding="utf-8")
    assert "maxIterations: 4," in src
    assert 'termination: "brief covered",' in src


@pytest.mark.parametrize("runtime", sorted(ADAPTERS))
def test_mcp_helper_ships_even_with_no_mcp_binding(runtime: str, tmp_path: Path) -> None:
    """``Node.tools`` are MCP tool names, so a loop that declares any needs
    the helper beside it — even when no node carries an ``mcp:`` binding.

    Missing this is a runtime failure inside the agent rather than an
    emit-time one, which is exactly the kind of gap worth closing here.
    """
    _emit(runtime, tmp_path)
    assert (tmp_path / "src" / "extracted" / "_roteMcp.ts").is_file()


@pytest.mark.parametrize("runtime", sorted(ADAPTERS))
def test_inference_helper_is_emitted_verbatim(runtime: str, tmp_path: Path) -> None:
    """Same contract as ``_roteMcp.ts``: never hand-edit the emitted copy.

    Byte equality is the test because a "small fix" applied to an emitted
    file instead of the constant is invisible until the next re-emit
    silently reverts it.
    """
    _emit(runtime, tmp_path)
    emitted = (tmp_path / "src" / "signatures" / "_roteInference.ts").read_text(encoding="utf-8")
    assert emitted == ROTE_INFERENCE_HELPER_TS


@pytest.mark.parametrize("runtime", sorted(ADAPTERS))
def test_unresolvable_servers_are_reported_at_emit_time(runtime: str, tmp_path: Path) -> None:
    """Bare tool names with nothing to resolve them against say so.

    The alternative is a loop that fails at run time with "no server
    provides this tool" and no hint that nothing was ever searched.
    """
    _emit(runtime, tmp_path)
    src = (tmp_path / "src" / "extracted" / "research_loop.ts").read_text(encoding="utf-8")
    assert "const SERVERS: string[] = [];" in src
    assert "no MCP server could be resolved at emit time" in src
    assert "ROTE_MCP_SERVERS" in src


@pytest.mark.parametrize("runtime", sorted(ADAPTERS))
def test_a_bound_sibling_supplies_the_search_list(runtime: str, tmp_path: Path) -> None:
    """When the pipeline knows a server, the loop searches it by name.

    This is the bridge until the compiler resolves tool → server at
    compile time; the servers a pipeline already talks to are the only
    honest guess available at emit time.
    """
    _emit(runtime, tmp_path, with_binding=True)
    src = (tmp_path / "src" / "extracted" / "research_loop.ts").read_text(encoding="utf-8")
    assert 'const SERVERS: string[] = ["research_tools"];' in src
    assert '"research_tools": "https://mcp.example.com/sse"' in src
    assert "no MCP server could be resolved at emit time" not in src


@pytest.mark.parametrize("runtime", sorted(ADAPTERS))
def test_anthropic_sdk_is_pinned_new_enough_for_the_tool_runner(
    runtime: str, tmp_path: Path
) -> None:
    """``beta.messages.toolRunner`` postdates the judge-era pins (0.91 / 0.110)."""
    _emit(runtime, tmp_path)
    deps = json.loads((tmp_path / "package.json").read_text())["dependencies"]
    assert deps["@anthropic-ai/sdk"] == "^0.115.0"


# ───────── Where the runtimes legitimately differ ─────────


def test_only_cloudflare_offers_the_workers_ai_lane(tmp_path: Path) -> None:
    """The lane is Cloudflare-only, and the Node runtimes must not even
    carry the dependency — a lane that worked on two of three runtimes
    would be a portability trap."""
    cf = tmp_path / "cf"
    _emit("cloudflare", cf)
    cf_src = (cf / "src" / "extracted" / "research_loop.ts").read_text(encoding="utf-8")
    assert "workersAi," in cf_src
    assert "@cloudflare/ai-utils" in cf_src
    assert "@cloudflare/ai-utils" in json.loads((cf / "package.json").read_text())["dependencies"]

    for runtime in ("dbos-ts", "inngest"):
        out = tmp_path / runtime
        _emit(runtime, out)
        src = (out / "src" / "extracted" / "research_loop.ts").read_text(encoding="utf-8")
        assert "workersAi" not in src
        assert "ai-utils" not in src
        assert "ai-utils" not in (out / "package.json").read_text()


def test_cloudflare_binds_ai_so_the_lane_is_reachable(tmp_path: Path) -> None:
    """Offered, never forced: the binding costs nothing until a run picks
    the lane, but without it "use my own Cloudflare account" is not an
    option the operator can take at all."""
    _emit("cloudflare", tmp_path)
    wrangler = (tmp_path / "wrangler.jsonc").read_text(encoding="utf-8")
    assert '"ai"' in wrangler
    assert '"binding": "AI"' in wrangler
    # Optional in Env — the loop runs on any other lane without it.
    assert "AI?: Ai;" in (tmp_path / "src" / "workflow.ts").read_text(encoding="utf-8")


def test_cloudflare_agent_loop_parks_on_auth(tmp_path: Path) -> None:
    """An agent loop's tools are MCP tools, so it parks like a bound step.

    The release event is derived from the failure rather than baked in:
    one loop may reach several servers, and ROTE_MCP_SERVERS can add more
    at run time.
    """
    _emit("cloudflare", tmp_path)
    src = (tmp_path / "src" / "workflow.ts").read_text(encoding="utf-8")
    assert "research_loop (agent loop): park on dead credentials" in src
    assert "authEventType(err," in src
    assert "NonRetryableError" in src


def test_cloudflare_agent_loop_stays_out_of_parallel_waves(tmp_path: Path) -> None:
    """A parkable step must not sit inside ``Promise.all``.

    ``waitForEvent``'s behavior inside a promise combinator is
    undocumented, and its timeout THROWS — which would reject every
    sibling in the wave.
    """
    pipeline = Pipeline.model_validate(
        {
            "name": "loop-plus-sibling",
            "version": "0.1.0",
            "source_skill": "tests/fixtures/loop-plus-sibling",
            "description": "An agent loop sharing a wave with a plain node.",
            "input": {"type": "Brief", "required": ["topic"], "optional": []},
            "nodes": [
                {
                    "id": "research_loop",
                    "kind": "agent_loop",
                    "description": "Research the topic.",
                    "tools": ["web_search"],
                    "inputs": {"topic": "pipeline.input.topic"},
                },
                {
                    "id": "taxonomy_lookup",
                    "kind": "pure_function",
                    "description": "Look up the taxonomy.",
                    "impl": "extracted/taxonomy.py:lookup",
                    "inputs": {"topic": "pipeline.input.topic"},
                },
            ],
            "edges": [],
            "entry_nodes": ["research_loop", "taxonomy_lookup"],
            "exit_nodes": ["research_loop", "taxonomy_lookup"],
        }
    )
    CloudflareAdapter().emit(pipeline, tmp_path)
    src = (tmp_path / "src" / "workflow.ts").read_text(encoding="utf-8")
    # The sibling is alone in the wave, so it emits sequentially; the loop
    # gets the retry/park wrapper. Neither appears inside a Promise.all.
    assert "Promise.all" not in src
    assert "park on dead credentials" in src
