"""Cross-runtime contract: a ``fan_out`` node dispatches once per element.

Every adapter must fan over the *same* input and hand each invocation a
single element. Until 0.12.x only DBOS did; the other five passed the
whole upstream list in one call, which meant the same ``pipeline.yaml``
produced two incompatible contracts for a user-filled stub — a per-post
judge on DBOS and a whole-post-list judge everywhere else. That is a
runtime leaking into node semantics, which invariant #1 forbids.

The assertions here are written to fail against batch dispatch, not just
to pass against the current output: each one checks that the element
param is bound to the loop variable AND that the whole-list expression
is *absent* from the payload. A test that only asserted "the list
expression appears somewhere" would pass either way, since per-element
dispatch iterates that same expression.
"""

from __future__ import annotations

import ast
import re

import pytest

from rote.adapters.cloudflare import emit_workflow as emit_cloudflare
from rote.adapters.dbos import emit_main as emit_dbos
from rote.adapters.dbos_ts import emit_main as emit_dbos_ts
from rote.adapters.inngest import emit_pipeline_ts as emit_inngest
from rote.adapters.python import emit_main as emit_python
from rote.adapters.temporal import emit_workflow as emit_temporal
from rote.ir import Edge, LLMSignature, Node, NodeKind, Pipeline

# ───────── Fixture: list-producer → fan_out judge ─────────
#
# Uses signature_spec (not the legacy Python-path form) so the identical
# pipeline is emittable by all six adapters — the whole point being a
# comparison across runtimes.

_SPEC = LLMSignature(
    input_schema={
        "type": "object",
        "properties": {"post": {"type": "string"}, "brief": {"type": "string"}},
        "required": ["post"],
    },
    output_schema={
        "type": "object",
        "properties": {"score": {"type": "number"}},
        "required": ["score"],
    },
    prompt="Score this post: {{ post }}",
)


def _fan_out_pipeline() -> Pipeline:
    return Pipeline(
        name="fan",
        input={"type": "In", "required": [], "optional": []},
        nodes=[
            Node(
                id="filter_posts",
                kind=NodeKind.PURE_FUNCTION,
                description="produce the list",
                impl="filters.py:filter_posts",
                inputs={"raw": "pipeline.input.raw"},
            ),
            Node(
                id="judge_content",
                kind=NodeKind.LLM_JUDGE,
                description="score ONE post",
                signature_spec=_SPEC,
                fan_out=True,
                inputs={
                    # the list to fan over
                    "post": "filter_posts.output.published_posts",
                    # shared by every invocation
                    "brief": "pipeline.input.brief",
                },
            ),
        ],
        edges=[{"from": "filter_posts", "to": "judge_content", "fan_out": True}],
        entry_nodes=["filter_posts"],
        exit_nodes=["judge_content"],
    )


# The expression each language uses for the whole upstream list. If this
# appears as the *value bound to the element param*, the adapter is
# batching.
_PY_LIST_EXPR = 'filter_posts_result["published_posts"]'
_TS_LIST_EXPR = '(filter_posts_result as Record<string, unknown>)["published_posts"]'

_EMITTERS = {
    "python": (emit_python, "py"),
    "temporal": (emit_temporal, "py"),
    "dbos": (emit_dbos, "py"),
    "cloudflare": (emit_cloudflare, "ts"),
    "dbos-ts": (emit_dbos_ts, "ts"),
    "inngest": (emit_inngest, "ts"),
}


@pytest.fixture(scope="module")
def emitted() -> dict[str, tuple[str, str]]:
    """``{runtime: (source, language)}`` for one shared fan_out pipeline."""
    pipeline = _fan_out_pipeline()
    return {name: (fn(pipeline), lang) for name, (fn, lang) in _EMITTERS.items()}


@pytest.mark.parametrize("runtime", sorted(_EMITTERS))
def test_element_param_binds_one_element_not_the_whole_list(
    runtime: str, emitted: dict[str, tuple[str, str]]
) -> None:
    """The judge receives ONE post, on every runtime.

    The negative half is what makes this test worth having: batch
    dispatch binds `post` straight to the list expression, and that is
    exactly the string asserted absent.
    """
    src, lang = emitted[runtime]
    if lang == "py":
        bound_to_element = '"post": _item' in src
        bound_to_list = f'"post": {_PY_LIST_EXPR}' in src
    else:
        bound_to_element = "post: _item" in src
        bound_to_list = f"post: {_TS_LIST_EXPR}" in src

    assert bound_to_element, (
        f"{runtime}: fan_out node's element param 'post' is not bound to the "
        f"per-element loop variable — it must receive one element, not the batch"
    )
    assert not bound_to_list, (
        f"{runtime}: fan_out node's element param 'post' is bound to the whole "
        f"upstream list; this is the batch dispatch fan_out is supposed to replace"
    )


@pytest.mark.parametrize("runtime", sorted(_EMITTERS))
def test_non_element_inputs_are_shared_by_every_invocation(
    runtime: str, emitted: dict[str, tuple[str, str]]
) -> None:
    """`brief` is not the fanned list, so every element gets it verbatim."""
    src, lang = emitted[runtime]
    expected = (
        '"brief": pipeline_input["brief"]' if lang == "py" else 'brief: pipelineInput["brief"]'
    )
    assert expected in src, f"{runtime}: shared input 'brief' missing from the fan_out payload"


@pytest.mark.parametrize("runtime", sorted(_EMITTERS))
def test_the_bound_list_is_iterated(runtime: str, emitted: dict[str, tuple[str, str]]) -> None:
    """Per-element dispatch means iterating the list, not passing it."""
    src, lang = emitted[runtime]
    # The guard call wraps against each adapter's own nesting depth, so
    # match the shape rather than a fixed indent.
    if lang == "py":
        pattern = r"for _item in _fan_out_list\(\s*" + re.escape(_PY_LIST_EXPR) + ","
        iterated = re.search(pattern, src) is not None
    else:
        pattern = r"fanOutList\(\s*" + re.escape(_TS_LIST_EXPR) + ","
        iterated = re.search(pattern, src) is not None and (").map(" in src or ").entries()" in src)
    assert iterated, f"{runtime}: the fanned list is never iterated per element"


@pytest.mark.parametrize("runtime", sorted(_EMITTERS))
def test_a_missing_upstream_key_names_the_node(
    runtime: str, emitted: dict[str, tuple[str, str]]
) -> None:
    """Every runtime reports a missing fanned key the same way.

    Regression for a live failure: the first real fan_out run on
    Cloudflare died with "Cannot read properties of undefined (reading
    'map')" — no node id, no reference, inside generated code the user
    never wrote. Python's bare "'NoneType' object is not iterable" is
    no better. Diagnostics are part of the emitted contract, so a guard
    on one runtime and not another is its own parity gap.
    """
    src, lang = emitted[runtime]
    helper, message = (
        ("def _fan_out_list(", "expected a list from")
        if lang == "py"
        else ("function fanOutList(", "expected an array from")
    )
    assert helper in src, f"{runtime}: fan_out guard helper not emitted"
    assert message in src, f"{runtime}: guard does not explain what it expected"
    # The guard must name the node and the IR reference it came from.
    assert "judge_content" in src and "filter_posts.output.published_posts" in src


@pytest.mark.parametrize("runtime", ["cloudflare", "inngest"])
def test_per_element_step_names_are_unique(
    runtime: str, emitted: dict[str, tuple[str, str]]
) -> None:
    """Cloudflare and Inngest key a durable step by its name.

    Reusing one name for every element is the nastiest possible failure
    here: Cloudflare returns the first element's cached result for all
    of them, so the run *succeeds* with silently uniform output. DBOS is
    exempt — it identifies a step by execution order, not by name.
    """
    src, _ = emitted[runtime]
    assert "judge_content[${_index}]" in src, (
        f"{runtime}: per-element steps must carry the element index in their "
        f"name; a constant name collapses the fan onto one cached step"
    )


@pytest.mark.parametrize("runtime", sorted(_EMITTERS))
def test_fanned_result_is_a_list(runtime: str, emitted: dict[str, tuple[str, str]]) -> None:
    """Downstream nodes see one result per element, in input order."""
    src, lang = emitted[runtime]
    if lang == "py":
        # list(pool.map(...)) / list(await asyncio.gather(...)) /
        # [_h.get_result() for _h in ...]
        assert re.search(r"judge_content_result = (list\(|\[)", src), (
            f"{runtime}: fanned result must bind as a list of per-element results"
        )
    else:
        array_forms = (
            r"await Promise\.all\(",  # cloudflare (plain), inngest
            r"judge_content_settled\.map\(",  # dbos-ts (allSettled + unwrap)
            r"judge_content_results",  # cloudflare (parkable, sequential)
        )
        pattern = r"judge_content_result = (" + "|".join(array_forms) + ")"
        assert re.search(pattern, src), (
            f"{runtime}: fanned result must bind as an array of per-element results"
        )


@pytest.mark.parametrize("runtime", ["python", "temporal", "dbos"])
def test_emitted_python_parses(runtime: str, emitted: dict[str, tuple[str, str]]) -> None:
    """The fan_out emission is syntactically valid Python."""
    src, _ = emitted[runtime]
    ast.parse(src)


def test_all_six_runtimes_agree_on_which_input_is_the_list() -> None:
    """The fanned input is a property of the IR, not of the target.

    Resolution lives in the language-neutral common module precisely so
    two adapters cannot disagree about which input is the list — that
    would make one pipeline mean two different things.
    """
    from rote.adapters._common import fan_out_element_param

    pipeline = _fan_out_pipeline()
    judge = pipeline.nodes[1]
    assert fan_out_element_param(judge, pipeline) == "post"


def test_ambiguous_fan_out_is_an_emit_time_error() -> None:
    """Two edge-fed node-bound inputs and no marker → refuse, never guess.

    Picking the wrong list silently judges the wrong things, which is
    far worse than failing to emit.
    """
    from rote.adapters._common import fan_out_element_param

    pipeline = Pipeline(
        name="ambiguous",
        input={"type": "In", "required": [], "optional": []},
        nodes=[
            Node(id="a", kind=NodeKind.PURE_FUNCTION, description="d", impl="m.py:a"),
            Node(id="b", kind=NodeKind.PURE_FUNCTION, description="d", impl="m.py:b"),
            Node(
                id="j",
                kind=NodeKind.LLM_JUDGE,
                description="d",
                signature_spec=_SPEC,
                fan_out=True,
                inputs={"post": "a.output.xs", "brief": "b.output.ys"},
            ),
        ],
        edges=[{"from": "a", "to": "j"}, {"from": "b", "to": "j"}],
        entry_nodes=["a", "b"],
        exit_nodes=["j"],
    )
    with pytest.raises(ValueError, match="cannot identify the element param"):
        fan_out_element_param(pipeline.nodes[2], pipeline)


def test_the_fan_out_edge_marker_beats_a_plain_edge() -> None:
    """Two edge-fed inputs, one edge marked `fan_out: true` — the marker wins.

    The precedence tiers only differ when more than one node-bound input
    is edge-fed, so a single-edge fixture exercises tier 1 and tier 2
    identically and cannot tell them apart. Found by mutation testing:
    swapping the tier order left the whole suite green, because nothing
    covered the case the ordering exists for.

    Getting this wrong fans over the wrong upstream list, which judges
    the wrong things silently rather than failing.
    """
    from rote.adapters._common import fan_out_element_param

    judge = Node(
        id="j",
        kind=NodeKind.LLM_JUDGE,
        description="d",
        signature_spec=_SPEC,
        fan_out=True,
        inputs={"post": "posts.output.items", "brief": "briefs.output.items"},
    )
    pipeline = Pipeline(
        name="two-edges",
        input={"type": "In", "required": [], "optional": []},
        nodes=[
            Node(id="posts", kind=NodeKind.PURE_FUNCTION, description="d", impl="m.py:posts"),
            Node(id="briefs", kind=NodeKind.PURE_FUNCTION, description="d", impl="m.py:briefs"),
            judge,
        ],
        # BOTH feed the judge; only `posts` is marked as the fanned list.
        edges=[
            {"from": "posts", "to": "j", "fan_out": True},
            {"from": "briefs", "to": "j"},
        ],
        entry_nodes=["posts", "briefs"],
        exit_nodes=["j"],
    )
    assert fan_out_element_param(judge, pipeline) == "post"

    # And symmetrically, so the test cannot pass by dict/sort order:
    # move the marker to the other edge and the answer must follow it.
    flipped = pipeline.model_copy(
        update={
            "edges": [
                Edge.model_validate({"from": "posts", "to": "j"}),
                Edge.model_validate({"from": "briefs", "to": "j", "fan_out": True}),
            ]
        }
    )
    assert fan_out_element_param(judge, flipped) == "brief"
