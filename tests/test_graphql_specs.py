"""GraphQL requests are checked against the vendor's own schema.

The schemas are AniList's, as its endpoint describes itself (tests/fixtures/
vendor_specs/anilist.graphql), and GitHub's public SDL with its prose
dropped (github.graphql); nothing here reaches the network.
``correct_calls/github_graphql`` and ``correct_calls/anilist`` are
requests written the way each vendor's docs say: every one must pass.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

import pytest

from rote.ir import load_pipeline
from rote.vendor_specs import SpecStore, check_deliverable
from rote.vendor_specs.graphql import REGISTRY, operation_type

FIXTURES = Path(__file__).parent / "fixtures" / "vendor_specs"
SDL = (FIXTURES / "anilist.graphql").read_text()
GITHUB_SDL = (FIXTURES / "github.graphql").read_text()
ANILIST = REGISTRY["https://graphql.anilist.co"]
GITHUB = REGISTRY["https://api.github.com/graphql"]
CORRECT_CALLS = FIXTURES / "correct_calls"


def anilist_store(tmp_path: Path) -> SpecStore:
    def introspect(endpoint: str) -> Any:
        raise AssertionError("the fixture is SDL; nothing introspects")

    store = SpecStore(root=tmp_path / "specs", introspect=introspect)
    # The cached document is what a real introspection would have stored.
    store._write("anilist/graphql/v2", {"source_url": ANILIST.endpoint, "document": SDL})
    return store


@pytest.fixture
def store(tmp_path: Path) -> SpecStore:
    return anilist_store(tmp_path)


def both_store(tmp_path: Path) -> SpecStore:
    """GitHub's SDL served as its file; AniList's endpoint introspected."""
    from graphql import build_schema, graphql_sync

    from rote.vendor_specs.graphql import INTROSPECTION_QUERY

    def fetch_text(url: str) -> str:
        assert url == GITHUB.sdl_url
        return GITHUB_SDL

    def introspect(endpoint: str) -> Any:
        assert endpoint == ANILIST.endpoint
        return {"data": graphql_sync(build_schema(SDL), INTROSPECTION_QUERY).data}

    return SpecStore(root=tmp_path / "specs", fetch_text=fetch_text, introspect=introspect)


PIPELINE = """\
name: season
version: "0.1.0"
description: d
input: {type: In}
nodes:
  - id: fetch_season
    kind: external_call
    description: Query AniList.
    impl: extracted/anilist.py:fetch_season
    inputs: {request: pipeline.input}
edges: []
entry_nodes: [fetch_season]
exit_nodes: [fetch_season]
"""

MODULE = '''
import httpx

QUERY = """
query ($season: MediaSeason, $year: Int, $perPage: Int) {
  Page(perPage: $perPage) {
    media(season: $season, seasonYear: $year, type: ANIME, sort: POPULARITY_DESC) {
      title { romaji }
      averageScore
      episodes
    }
  }
}
"""


def fetch_season(request):
    variables = {"season": request["season"], "year": request["year"], "perPage": 5}
    body = httpx.post(
        "https://graphql.anilist.co", json={"query": QUERY, "variables": variables}
    ).json()
    rows = []
    for m in body["data"]["Page"]["media"]:
        rows.append({"title": m["title"]["romaji"], "score": m.get("averageScore")})
    return {"rows": rows}
'''


def _check(tmp_path: Path, store: SpecStore, module: str) -> Any:
    root = tmp_path / "d"
    (root / "extracted").mkdir(parents=True)
    (root / "pipeline.yaml").write_text(PIPELINE, encoding="utf-8")
    (root / "extracted" / "anilist.py").write_text(textwrap.dedent(module), encoding="utf-8")
    [check] = check_deliverable(load_pipeline(root / "pipeline.yaml"), root, store)
    return check


def test_a_valid_query_passes_and_is_a_read(tmp_path: Path, store: SpecStore) -> None:
    check = _check(tmp_path, store, MODULE)
    assert (check.status, check.operation, check.writes) == ("ok", "graphql query", False)
    assert check.spec == "anilist/graphql/v2 (GraphQL schema)"


@pytest.mark.parametrize(
    ("old", "new", "kind", "says"),
    [
        ("averageScore\n", "avgScore\n", "graphql_invalid", "Cannot query field 'avgScore'"),
        ("sort: POPULARITY_DESC", "sort: MOST_POPULAR", "graphql_invalid", "MOST_POPULAR"),
        ('"perPage": 5}', '"perPage": 5, "page": 2}', "graphql_unknown_variable", "$page"),
        (
            'm.get("averageScore")',
            'm.get("meanScore")',
            "spec_unknown_response_field",
            "data.Page.media.meanScore",
        ),
        ("query ($season", "mutation ($season", "graphql_invalid", None),
    ],
)
def test_each_graphql_problem_is_found(
    tmp_path: Path, store: SpecStore, old: str, new: str, kind: str, says: str | None
) -> None:
    assert old in MODULE
    check = _check(tmp_path, store, MODULE.replace(old, new, 1))
    kinds = [f.kind for f in check.findings]
    assert kind in kinds
    if says:
        assert any(says in f.message for f in check.findings if f.kind == kind)


def test_a_required_variable_that_is_not_sent_is_found(tmp_path: Path, store: SpecStore) -> None:
    module = MODULE.replace("$season: MediaSeason,", "$season: MediaSeason!,")
    module = module.replace('"season": request["season"], ', "")
    check = _check(tmp_path, store, module)
    assert [f.kind for f in check.findings] == ["graphql_missing_variable"]


def test_a_document_built_at_run_time_is_listed_not_guessed(
    tmp_path: Path, store: SpecStore
) -> None:
    module = MODULE.replace('json={"query": QUERY,', 'json={"query": QUERY.replace("5", "6"),')
    check = _check(tmp_path, store, module)
    assert check.status == "unresolved" and "string literal" in (check.note or "")


def test_operation_type_reads_the_document() -> None:
    assert operation_type("{ Viewer { id } }") == "query"
    assert operation_type("mutation { ToggleLike(id: 1) { __typename } }") == "mutation"
    assert operation_type("not graphql") is None


def test_a_valid_mutation_is_a_write_and_the_live_check_refuses_it(
    tmp_path: Path, store: SpecStore
) -> None:
    from rote.compiler.backend import write_backend
    from rote.vendor_specs.fixture_store import FixtureStore
    from rote.vendor_specs.live import LiveCheckRefused, live_check

    module = """
import httpx

FOLLOW = "mutation ($id: Int) { ToggleFollow(userId: $id) { id } }"


def fetch_season(request):
    return httpx.post(
        "https://graphql.anilist.co", json={"query": FOLLOW, "variables": {"id": 1}}
    ).json()
"""
    check = _check(tmp_path, store, module)
    assert (check.status, check.operation, check.writes) == ("ok", "graphql mutation", True)
    root = tmp_path / "d"
    write_backend(root, "api")
    with pytest.raises(LiveCheckRefused, match="graphql mutation"):
        live_check(root, {}, store, FixtureStore(tmp_path / "fx"), confirm=lambda plan: True)


# ── every vendor's real calls, and GitHub's real schema ──


def _run(tmp_path: Path, store: SpecStore, module: str) -> list[Any]:
    root = tmp_path / "c"
    (root / "extracted").mkdir(parents=True)
    (root / "pipeline.yaml").write_text(PIPELINE, encoding="utf-8")
    source = textwrap.dedent(module).replace("def f(", "def fetch_season(")
    (root / "extracted" / "anilist.py").write_text(source, encoding="utf-8")
    return check_deliverable(load_pipeline(root / "pipeline.yaml"), root, store)


def test_githubs_real_schema_builds_and_checks_a_request(tmp_path: Path) -> None:
    """GitHub's public schema fails graphql-core's schema validation (a
    deprecated field implements one its interface does not deprecate);
    a client's check must not depend on that."""
    from graphql import build_schema, validate_schema

    assert validate_schema(build_schema(GITHUB_SDL))  # the vendor's own schema errors
    module = (CORRECT_CALLS / "github_graphql" / "issues.py").read_text(encoding="utf-8")
    [check] = _run(tmp_path, both_store(tmp_path), module)
    assert (check.status, check.operation, check.writes) == ("ok", "graphql query", False)
    assert check.spec is not None and check.spec.startswith("github/graphql/v4")


#: Correct calls whose reads go below a custom scalar: AniList's
#: advancedScores is Json, which the schema does not describe.
PARTIAL = {("anilist", "lists.py")}


@pytest.mark.parametrize(
    "module",
    sorted(p for d in ("github_graphql", "anilist") for p in (CORRECT_CALLS / d).glob("*.py")),
    ids=lambda p: f"{p.parent.name}/{p.name}",
)
def test_correct_real_world_graphql_calls_report_no_findings(tmp_path: Path, module: Path) -> None:
    [check] = _run(tmp_path, both_store(tmp_path), module.read_text(encoding="utf-8"))
    assert check.findings == [], [f.message for f in check.findings]
    expected = "partial" if (module.parent.name, module.name) in PARTIAL else "ok"
    assert check.status == expected, (check.note, check.unchecked)
    writes = module.name == "comment.py"
    assert check.operation == ("graphql mutation" if writes else "graphql query")
    assert check.writes is writes


@pytest.mark.parametrize(
    ("module", "old", "new", "kinds"),
    [
        ("github_graphql/issues.py", "totalCount\n", "count\n", ["graphql_invalid"]),
        ("github_graphql/issues.py", "states: OPEN", "states: OPENED", ["graphql_invalid"]),
        (
            "github_graphql/issues.py",
            '"cursor": None}',
            '"cursor": None, "x": 1}',
            ["graphql_unknown_variable"],
        ),
        (
            "github_graphql/issues.py",
            '"owner": request["owner"], ',
            "",
            ["graphql_missing_variable"],
        ),
        ("github_graphql/issues.py", 'i["url"]', 'i["html_url"]', ["spec_unknown_response_field"]),
        # A String has no fields: reading into one is a bug, not a gap.
        (
            "github_graphql/issues.py",
            'i["title"]',
            'i["title"]["text"]',
            ["spec_unknown_response_field"],
        ),
        # Inline fragments and named fragments are followed.
        ("github_graphql/search.py", 'n["url"]', 'n["permalink"]', ["spec_unknown_response_field"]),
        ("github_graphql/fragments.py", '["oid"]', '["sha"]', ["spec_unknown_response_field"]),
        (
            "github_graphql/fragments.py",
            'r["stargazerCount"]',
            'r["stars"]',
            ["spec_unknown_response_field"],
        ),
        # Several operations: the request must name one, and one that exists.
        ("github_graphql/search.py", '"operationName": "Search", ', "", ["graphql_invalid"]),
        (
            "github_graphql/search.py",
            '"operationName": "Search"',
            '"operationName": "Find"',
            ["graphql_invalid"],
        ),
        # A literal variable of the wrong type.
        ("anilist/season.py", '"year": 2026', '"year": "2026"', ["graphql_variable_value"]),
        (
            "anilist/season.py",
            '"season": "WINTER"',
            '"season": "WINTERS"',
            ["graphql_variable_value"],
        ),
    ],
)
def test_a_wrong_graphql_call_is_caught(
    tmp_path: Path, module: str, old: str, new: str, kinds: list[str]
) -> None:
    source = (CORRECT_CALLS / module).read_text(encoding="utf-8")
    assert source.count(old) == 1, old
    [check] = _run(tmp_path, both_store(tmp_path), source.replace(old, new))
    assert [f.kind for f in check.findings] == kinds, [f.message for f in check.findings]


def test_a_read_below_a_custom_scalar_is_named_not_flagged(tmp_path: Path) -> None:
    module = (CORRECT_CALLS / "anilist" / "lists.py").read_text(encoding="utf-8")
    [check] = _run(tmp_path, both_store(tmp_path), module)
    assert check.unchecked == [
        "line 32 reads data.MediaListCollection.lists.entries.advancedScores.Story: "
        "MediaListCollection.lists.entries.advancedScores is the scalar Json, which the "
        "schema does not describe"
    ]


def test_a_computed_operation_name_is_not_checked(tmp_path: Path) -> None:
    source = (CORRECT_CALLS / "github_graphql" / "search.py").read_text(encoding="utf-8")
    source = source.replace('"operationName": "Search"', '"operationName": request["op"]')
    [check] = _run(tmp_path, both_store(tmp_path), source)
    assert (check.status, check.findings) == ("partial", [])
    assert check.unchecked == [
        "the operation: the document has several, and operationName is computed"
    ]


@pytest.mark.parametrize(
    "module",
    [
        # Rebound: the request could send either document.
        """
        import httpx
        Q = "query { Viewer { id } }"
        Q = "mutation { ToggleFollow(userId: 1) { id } }"
        def f(request):
            return httpx.post("https://graphql.anilist.co", json={"query": Q}).json()
        """,
        # Bound under a condition: the mutation is sent when it holds.
        """
        import httpx
        def f(request):
            Q = "query { Viewer { id } }"
            if request["follow"]:
                Q = "mutation { ToggleFollow(userId: 1) { id } }"
            return httpx.post("https://graphql.anilist.co", json={"query": Q}).json()
        """,
        # The function's own Q hides the module's.
        """
        import httpx
        Q = "query { Viewer { id } }"
        def f(request):
            if request["follow"]:
                Q = "mutation { ToggleFollow(userId: 1) { id } }"
            return httpx.post("https://graphql.anilist.co", json={"query": Q}).json()
        """,
        # A parameter named like the module's document is the caller's.
        """
        import httpx
        Q = "query { Viewer { id } }"
        def f(Q):
            return httpx.post("https://graphql.anilist.co", json={"query": Q}).json()
        """,
        # A body dict rebound.
        """
        import httpx
        def f(request):
            body = {"query": "query { Viewer { id } }"}
            body = {"query": "mutation { ToggleFollow(userId: 1) { id } }"}
            return httpx.post("https://graphql.anilist.co", json=body).json()
        """,
    ],
)
def test_a_document_that_could_be_another_is_not_checked(tmp_path: Path, module: str) -> None:
    [check] = _run(tmp_path, both_store(tmp_path), module)
    assert check.status == "unresolved" and check.operation is None


def test_a_body_dict_bound_once_is_followed(tmp_path: Path) -> None:
    module = """
    import httpx
    def f(request):
        payload = {"query": "query { Viewer { id } }", "variables": {"x": 1}}
        return httpx.post("https://graphql.anilist.co", json=payload).json()
    """
    [check] = _run(tmp_path, both_store(tmp_path), module)
    assert [f.kind for f in check.findings] == ["graphql_unknown_variable"]


def test_a_document_bound_once_is_read_wherever_it_is_used(tmp_path: Path) -> None:
    module = """
    import httpx
    Q = "query { Viewer { id name } }"
    def f(request):
        a = httpx.post("https://graphql.anilist.co", json={"query": Q}).json()
        return a["data"]["Viewer"]["name"]
    """
    [check] = _run(tmp_path, both_store(tmp_path), module)
    assert (check.status, check.operation) == ("ok", "graphql query")


def test_a_graphql_get_is_listed_not_sent_to_the_rest_check(tmp_path: Path) -> None:
    module = """
    import httpx
    def f(request):
        return httpx.get("https://api.github.com/graphql", params={"query": "{ viewer { login } }"})
    """
    [check] = _run(tmp_path, both_store(tmp_path), module)
    assert (check.status, check.findings) == ("unresolved", [])
    assert "GraphQL document sent as a POST body" in (check.note or "")


def test_the_schema_revision_is_its_content_not_its_length() -> None:
    from rote.vendor_specs.graphql import to_api_spec

    a = to_api_spec("type Query { a: Int }", ANILIST)
    b = to_api_spec("type Query { b: Int }", ANILIST)
    assert len("type Query { a: Int }") == len("type Query { b: Int }")
    assert a.revision != b.revision


def test_a_schema_that_does_not_build_is_not_cached(tmp_path: Path) -> None:
    from rote.vendor_specs import SpecUnavailable

    root = tmp_path / "specs"
    broken = SpecStore(root=root, fetch_text=lambda url: "type Query { a: Nope }")
    with pytest.raises(SpecUnavailable, match="not a GraphQL schema rote can build"):
        broken.graphql(GITHUB)
    calls: list[str] = []

    def fetch_text(url: str) -> str:
        calls.append(url)
        return GITHUB_SDL

    assert SpecStore(root=root, fetch_text=fetch_text).graphql(GITHUB).key == "github/graphql/v4"
    assert calls == [GITHUB.sdl_url]


@pytest.mark.parametrize(
    ("body", "kinds", "says"),
    [
        # Does not parse.
        ('Q = "query { Viewer { id }"\nr = post(Q, {})', ["graphql_invalid"], "does not parse"),
        # A required variable with a default, or variables the check cannot
        # see, is not missing.
        ('Q = "query ($id: Int! = 1) { User(id: $id) { id } }"\nr = post(Q, {})', [], None),
        (
            'Q = "query ($id: Int!) { User(id: $id) { id } }"\nr = post(Q, request["vars"])',
            [],
            None,
        ),
        # One unselected field is one finding, whatever reads below it.
        (
            'Q = "query { Viewer { id } }"\nr = post(Q, {})\n'
            'return r["data"]["Viewer"]["name"], r["data"]["Viewer"]["name"]["x"]',
            ["spec_unknown_response_field"],
            "data.Viewer.name, which the query does not select",
        ),
        # An alias is the response key.
        (
            'Q = "query { me: Viewer { id } }"\nr = post(Q, {})\nreturn r["data"]["me"]["id"]',
            [],
            None,
        ),
        (
            'Q = "query { me: Viewer { id } }"\nr = post(Q, {})\nreturn r["data"]["Viewer"]',
            ["spec_unknown_response_field"],
            "data.Viewer, which",
        ),
        # __typename is a String, and so is a mutation's field here.
        (
            'Q = "query { Viewer { __typename } }"\nr = post(Q, {})\n'
            'return r["data"]["Viewer"]["__typename"]["x"]',
            ["spec_unknown_response_field"],
            "Viewer.__typename is a String, which has no fields",
        ),
        (
            'Q = "mutation { UpdateUser(about: \\"x\\") { about } }"\nr = post(Q, {})\n'
            'return r["data"]["UpdateUser"]["about"]["x"]',
            ["spec_unknown_response_field"],
            "UpdateUser.about is a String, which has no fields",
        ),
    ],
)
def test_anilist_request_shapes(
    tmp_path: Path, body: str, kinds: list[str], says: str | None
) -> None:
    lines = body.splitlines()
    # `post(Q, vars)` in a case is the request, written out in f.
    module = "import httpx\n" + lines[0] + "\n\n\ndef f(request):\n"
    call = lines[1].replace(
        "post(Q, ",
        'httpx.post("https://graphql.anilist.co/", json={"query": Q, "variables": ',
    )
    call = call[: call.rindex(")")] + "}).json()"
    module += f"    {call}\n" + "".join(f"    {line}\n" for line in lines[2:])
    [check] = _run(tmp_path, both_store(tmp_path), module)
    assert [f.kind for f in check.findings] == kinds, [f.message for f in check.findings]
    if says:
        assert any(says in f.message for f in check.findings)
    # The endpoint is matched with its trailing slash.
    assert check.spec is not None and check.spec.startswith("anilist/graphql/v2")


def test_a_schema_is_cached_and_refetched_when_its_url_moves(tmp_path: Path) -> None:
    from dataclasses import replace

    calls: list[str] = []

    def fetch_text(url: str) -> str:
        calls.append(url)
        return GITHUB_SDL

    root = tmp_path / "specs"
    SpecStore(root=root, fetch_text=fetch_text).graphql(GITHUB)
    SpecStore(root=root, fetch_text=fetch_text).graphql(GITHUB)
    assert calls == [GITHUB.sdl_url]
    moved = replace(GITHUB, sdl_url="https://docs.github.com/public/ghec/schema.docs.graphql")
    SpecStore(root=root, fetch_text=fetch_text).graphql(moved)
    assert calls == [GITHUB.sdl_url, moved.sdl_url]


def test_the_live_check_builds_only_a_query(tmp_path: Path) -> None:
    from rote.vendor_specs.extract import extract
    from rote.vendor_specs.live import _build_graphql, _Unbuildable

    spec = both_store(tmp_path).graphql(ANILIST)
    [site] = extract(
        'import httpx\nM = "mutation { ToggleFollow(userId: 1) { id } }"\n'
        'def f():\n    httpx.post("https://graphql.anilist.co", json={"query": M})\n'
    )
    with pytest.raises(_Unbuildable, match="not exactly one query"):
        _build_graphql(site, spec, {}, {})


def test_a_live_graphql_read_reports_what_the_response_lacks(tmp_path: Path) -> None:
    """The live response, against the fields the code reads (the endpoint's
    spec found by its URL)."""
    import json

    from rote.compiler.backend import write_backend
    from rote.vendor_specs.fixture_store import FixtureStore
    from rote.vendor_specs.live import LiveRequest, live_check

    root = tmp_path / "d"
    (root / "extracted").mkdir(parents=True)
    (root / "pipeline.yaml").write_text(PIPELINE, encoding="utf-8")
    (root / "extracted" / "anilist.py").write_text(textwrap.dedent(MODULE), encoding="utf-8")
    write_backend(root, "api")
    sent: list[LiveRequest] = []

    def send(request: LiveRequest) -> Any:
        sent.append(request)
        media = [{"title": {"romaji": "x"}, "episodes": 12}]
        return 200, "application/json", json.dumps({"data": {"Page": {"media": media}}}).encode()

    report = live_check(
        root,
        {"season": "FALL", "year": 2026},
        anilist_store(tmp_path),
        FixtureStore(tmp_path / "fx"),
        env={},
        send=send,
        confirm=lambda plan: True,
    )
    [check] = report.checked
    assert len(sent) == 1
    assert check.reads_missing == ["data.Page.media[].averageScore"]


def test_a_graphql_read_sends_no_credential_the_spec_does_not_declare(tmp_path: Path) -> None:
    """GitHub's schema declares no credential, so the live check refuses to
    send the token the code reads, as for any REST read."""
    from rote.compiler.backend import write_backend
    from rote.vendor_specs.fixture_store import FixtureStore
    from rote.vendor_specs.live import CredentialRefused, live_check

    root = tmp_path / "c"
    _run(
        tmp_path, both_store(tmp_path), (CORRECT_CALLS / "github_graphql" / "issues.py").read_text()
    )
    write_backend(root, "api")
    sent: list[Any] = []
    with pytest.raises(CredentialRefused, match="GITHUB_TOKEN"):
        live_check(
            root,
            {"owner": "o", "repo": "r"},
            both_store(tmp_path / "2"),
            FixtureStore(tmp_path / "fx"),
            env={"GITHUB_TOKEN": "ghp-secret"},
            send=sent.append,
            confirm=lambda plan: True,
        )
    assert sent == []


def test_a_docs_spec_for_a_graphql_vendor_is_refused(tmp_path: Path) -> None:
    import json

    from rote.vendor_specs.docs import DocsSpecError, load

    root = tmp_path / "c"
    (root / "vendor_specs").mkdir(parents=True)
    (root / "vendor_specs" / "a.json").write_text(
        json.dumps(
            {
                "vendor": "anilist",
                "api": "graphql",
                "version": "v2",
                "sources": ["https://docs.anilist.co"],
                "operations": [{"id": "x", "method": "POST", "url_template": "https://x.example"}],
            }
        )
    )
    with pytest.raises(DocsSpecError, match="anilist publishes a spec rote reads"):
        load(root)
