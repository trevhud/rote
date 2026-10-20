"""`--system1-decider clef`: Clef's confident answers over the rules, nothing silent.

The HTTP call is stubbed (no inference in tests). What is pinned: the request
is exactly the pre-registered one, the thresholds route as measured, and a
missing credential fails the compile instead of quietly using the rules.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from rote.compiler import Compiler, CompilerError, system1
from rote.compiler.system1 import clef

STEPS = (
    "## Step 1: Total the batch\n\nTotal every line in USD.\n\n"
    "## Step 2: Tidy the notes\n\nTidy the notes for the file.\n\n"
    "## Step 3: Score the leads\n\nScore the leads against the rubric.\n"
)


class _NeverRuns:
    """A driver the compile must not reach: the decider fails first."""

    name = "never-runs"

    async def run(self, **kwargs: object) -> object:
        raise AssertionError("the agent ran although the decider could not")


def _skill(tmp_path: Path) -> Path:
    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text(f"---\nname: tiny\ndescription: x\n---\n\n# Tiny\n\n{STEPS}")
    return skill


def _answers(kind: dict[str, float], human: float) -> dict[str, Any]:
    return {
        "result": {
            "answers": {
                "kind": {"type": "choice", "probabilities": kind},
                "human": {"type": "noul", "noul": human},
            },
            "usage": {"input_tokens": 1000},
        }
    }


ZERO = dict.fromkeys(clef.KINDS["v0"], 0.0)


@pytest.fixture
def creds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct123")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "tok")
    monkeypatch.setattr(clef.time, "sleep", lambda _s: None)


def _stub(monkeypatch: pytest.MonkeyPatch, by_title: dict[str, Any]) -> list[dict]:
    calls: list[dict[str, Any]] = []

    def post(url: str, token: str, body: dict[str, Any]) -> dict[str, Any]:
        calls.append({"url": url, "token": token, "body": body})
        answer = by_title[body["state"]["this_step"]["title"]]
        if isinstance(answer, list):  # one answer per phrasing, v0..v2
            criteria = body["questions"]["kind"]["criteria"]
            variant = next(v for v, c in clef.KINDS.items() if c == criteria)
            return answer[int(variant[1])]
        return answer

    monkeypatch.setattr(clef, "_post", post)
    return calls


def test_the_request_is_the_pre_registered_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, creds: None
) -> None:
    calls = _stub(
        monkeypatch,
        {
            t: _answers({**ZERO, "pure_function": 0.5, "llm_judge": 0.5}, 0.1)
            for t in (
                "Step 1: Total the batch",
                "Step 2: Tidy the notes",
                "Step 3: Score the leads",
            )
        },
    )
    work = tmp_path / "work"
    work.mkdir()
    system1.prepare(_skill(tmp_path), work, decider="clef")
    assert len(calls) == 9  # 3 steps x 3 phrasings
    first = calls[0]
    assert first["url"] == (
        "https://api.cloudflare.com/client/v4/accounts/acct123/ai/run/@cf/cloudflare/clef"
    )
    assert first["token"] == "tok"
    body = first["body"]
    assert body["model"] == "clef"
    assert set(body["state"]) == {"skill", "skill_description", "all_steps", "this_step"}
    assert body["state"]["all_steps"] == [
        "Step 1: Total the batch",
        "Step 2: Tidy the notes",
        "Step 3: Score the leads",
    ]
    variants = [c["body"]["questions"] for c in calls[:3]]
    assert [v["kind"]["criteria"] for v in variants] == [clef.KINDS[k] for k in ("v0", "v1", "v2")]
    assert [v["human"]["instructions"] for v in variants] == [
        clef.HUMAN[k] for k in ("v0", "v1", "v2")
    ]
    assert clef.KIND_THRESHOLD == 0.85 and clef.HUMAN_THRESHOLD == 0.50


def test_confident_answers_win_and_unsure_ones_go_to_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, creds: None
) -> None:
    _stub(
        monkeypatch,
        {
            # The rules say pure_function; Clef is sure a person must act.
            "Step 1: Total the batch": _answers({**ZERO, "pure_function": 0.9}, 0.92),
            # No signal for the rules (pure_function, 0.3); Clef is sure it is a call.
            "Step 2: Tidy the notes": _answers({**ZERO, "external_call": 0.93}, 0.05),
            # The rules say llm_judge; Clef leans pure_function but below 0.85.
            # The rules say llm_judge. Clef's first phrasing is sure it is
            # pure_function (0.95), but the three-phrasing mean (0.78) is not.
            "Step 3: Score the leads": [
                _answers({**ZERO, "pure_function": 0.95, "llm_judge": 0.05}, 0.1),
                _answers({**ZERO, "pure_function": 0.70, "llm_judge": 0.30}, 0.1),
                _answers({**ZERO, "pure_function": 0.70, "llm_judge": 0.30}, 0.1),
            ],
        },
    )
    work = tmp_path / "work"
    work.mkdir()
    plan = system1.prepare(_skill(tmp_path), work, decider="clef")
    doc = yaml.safe_load((work / "pipeline.yaml").read_text())
    kinds = {n["source"]["section"]: n["kind"] for n in doc["nodes"]}
    assert kinds["Step 1: Total the batch"] == "hitl_gate"
    assert kinds["Step 2: Tidy the notes"] == "external_call"
    assert kinds["Step 3: Score the leads"] == "llm_judge"  # the rules decide below threshold
    assert any("Clef disagrees below its threshold" in r for r in plan.review)
    decisions = (work / "system1" / "decisions.json").read_text()
    assert '"model": "@cf/cloudflare/clef"' in decisions and '"cost_usd": 0.00216' in decisions


@pytest.mark.asyncio
async def test_asking_for_clef_without_credentials_fails_the_compile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
    monkeypatch.delenv("CLOUDFLARE_API_TOKEN", raising=False)

    def no_call(url: str, token: str, body: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError("Clef was called without credentials")

    monkeypatch.setattr(clef, "_post", no_call)
    monkeypatch.setattr(Compiler, "select_driver", lambda self: _NeverRuns())
    compiler = Compiler(system1=True, system1_decider="clef")
    with pytest.raises(CompilerError, match="CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN"):
        await compiler.compile(_skill(tmp_path), tmp_path / "out")


def test_a_rate_limit_is_retried_and_a_dead_endpoint_is_loud(
    monkeypatch: pytest.MonkeyPatch, creds: None
) -> None:
    attempts = {"n": 0}

    def flaky(url: str, token: str, body: dict[str, Any]) -> dict[str, Any]:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise clef.ClefUnavailable("Workers AI answered HTTP 429")
        return _answers({**ZERO, "pure_function": 1.0}, 0.0)

    monkeypatch.setattr(clef, "_post", flaky)
    assert clef.ask("a", "t", {}, "v0")["answers"]["human"]["noul"] == 0.0
    assert attempts["n"] == 3

    def dead(url: str, token: str, body: dict[str, Any]) -> dict[str, Any]:
        raise clef.ClefUnavailable("Workers AI answered HTTP 500")

    monkeypatch.setattr(clef, "_post", dead)
    with pytest.raises(clef.ClefUnavailable, match="did not answer after 6 tries"):
        clef.ask("a", "t", {}, "v0")


def test_a_refused_request_fails_at_once(monkeypatch: pytest.MonkeyPatch, creds: None) -> None:
    """A 401/403/404 fails the same way every time: no six rounds of backoff."""
    attempts = {"n": 0}

    def refused(url: str, token: str, body: dict[str, Any]) -> dict[str, Any]:
        attempts["n"] += 1
        raise clef.ClefRejected("Workers AI answered HTTP 401: b'bad token'")

    monkeypatch.setattr(clef, "_post", refused)
    with pytest.raises(clef.ClefRejected, match="HTTP 401"):
        clef.ask("a", "t", {}, "v0")
    assert attempts["n"] == 1


@pytest.mark.parametrize(
    ("code", "rejected"), [(401, True), (403, True), (404, True), (429, False), (500, False)]
)
def test_http_status_maps_to_retry_or_not(
    monkeypatch: pytest.MonkeyPatch, code: int, rejected: bool
) -> None:
    import io
    import urllib.error

    def raise_http(req: object, timeout: float) -> object:
        raise urllib.error.HTTPError("u", code, "m", {}, io.BytesIO(b"body"))  # type: ignore[arg-type]

    monkeypatch.setattr(clef.urllib.request, "urlopen", raise_http)
    with pytest.raises(clef.ClefUnavailable) as err:
        clef._post("https://example.invalid", "t", {})
    assert isinstance(err.value, clef.ClefRejected) is rejected


def test_the_clef_decider_needs_system1(tmp_path: Path) -> None:
    from rote import cli

    code = cli.main(
        [
            "compile",
            str(_skill(tmp_path)),
            "--out",
            str(tmp_path / "o"),
            "--no-system1",
            "--system1-decider",
            "clef",
        ]
    )
    assert code == 2


def _sent_headers(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Run one real _post against a captured urlopen; return the headers sent."""
    import io

    sent: dict[str, str] = {}

    def capture(req: Any, timeout: float) -> Any:
        sent.update({k.lower(): v for k, v in req.header_items()})
        return io.BytesIO(b'{"result": {}}')

    monkeypatch.setattr(clef.urllib.request, "urlopen", capture)
    clef._post("https://example.invalid", "t", {})
    return sent


def test_gateway_metadata_tags_every_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """A host meters Clef per compile by filtering gateway logs on this metadata."""
    monkeypatch.setenv("CLOUDFLARE_AI_GATEWAY_ID", "rote")
    monkeypatch.setenv("CLOUDFLARE_AI_GATEWAY_METADATA", '  {"graduation_id":   "g1"} ')
    sent = _sent_headers(monkeypatch)
    assert sent["cf-aig-gateway-id"] == "rote"
    # Re-serialized compactly, so stray whitespace in the env never reaches the header.
    assert sent["cf-aig-metadata"] == '{"graduation_id":"g1"}'


def test_no_gateway_settings_send_no_gateway_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CLOUDFLARE_AI_GATEWAY_ID", raising=False)
    monkeypatch.delenv("CLOUDFLARE_AI_GATEWAY_METADATA", raising=False)
    sent = _sent_headers(monkeypatch)
    assert "cf-aig-gateway-id" not in sent and "cf-aig-metadata" not in sent


@pytest.mark.parametrize(
    ("gateway", "metadata", "match"),
    [
        ("", '{"a": "b"}', "CLOUDFLARE_AI_GATEWAY_ID is not"),
        ("rote", "not json", "is not JSON"),
        ("rote", '["a"]', "must be a JSON object"),
        ("rote", json.dumps({f"k{i}": i for i in range(6)}), "keeps only 5"),
        ("rote", '{"cf.user_id": "u"}', "strips cf"),
        ("rote", '{"nested": {"a": 1}}', "string, number or boolean"),
        ("rote", '{"missing": null}', "string, number or boolean"),
    ],
)
def test_bad_gateway_metadata_fails_before_any_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    creds: None,
    gateway: str,
    metadata: str,
    match: str,
) -> None:
    """Untagged calls would bill the host with no compile to charge: refuse up front."""
    monkeypatch.setenv("CLOUDFLARE_AI_GATEWAY_ID", gateway)
    monkeypatch.setenv("CLOUDFLARE_AI_GATEWAY_METADATA", metadata)
    calls = _stub(monkeypatch, {})
    work = tmp_path / "work"
    work.mkdir()
    with pytest.raises(clef.ClefUnavailable, match=match):
        system1.prepare(_skill(tmp_path), work, decider="clef")
    assert calls == []


def test_five_scalar_entries_are_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    meta = {"a": "s", "b": 1, "c": 1.5, "d": True, "graduation_id": "g1"}
    monkeypatch.setenv("CLOUDFLARE_AI_GATEWAY_ID", "rote")
    monkeypatch.setenv("CLOUDFLARE_AI_GATEWAY_METADATA", json.dumps(meta))
    assert json.loads(_sent_headers(monkeypatch)["cf-aig-metadata"]) == meta
