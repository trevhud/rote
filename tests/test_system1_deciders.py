"""`--decider clef|jev`: a decision model's confident answers over the rules, nothing silent.

The HTTP call is stubbed (no inference in tests). What is pinned: each
model's request is exactly the pre-registered one on its own route, the
frozen thresholds route as measured, and a missing credential fails the
compile instead of quietly using the rules.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from rote.compiler import Compiler, CompilerError, system1
from rote.compiler.system1 import decision_model as dm
from rote.compiler.system1.decision_model import CLEF, JEV, DecisionModel

MODELS = pytest.mark.parametrize("model", [CLEF, JEV], ids=["clef", "jev"])
#: Jev's thresholds as fitted on the dev skills (docs/compile-bench.md).
JEV_KIND, JEV_HUMAN = 0.85, 0.60

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
    """A per-model route's response (Clef's shape)."""
    return {
        "result": {
            "answers": {
                "kind": {"type": "choice", "probabilities": kind},
                "human": {"type": "noul", "noul": human},
            },
            "usage": {"input_tokens": 1000},
        }
    }


def _as_served(model: DecisionModel, response: dict[str, Any]) -> dict[str, Any]:
    """The same answer as ``model``'s route returns it: the unified /ai/run
    route (Jev) wraps the record once more."""
    if model.selector is None:
        return {"result": {"result": response["result"]}}
    return response


def _request(body: dict[str, Any]) -> dict[str, Any]:
    """``{"state", "questions"}`` from either route's body."""
    return body.get("input", body)


ZERO = dict.fromkeys(dm.KINDS["v0"], 0.0)


@pytest.fixture
def creds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct123")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "tok")
    monkeypatch.setattr(dm.time, "sleep", lambda _s: None)


def _stub(
    monkeypatch: pytest.MonkeyPatch, by_title: dict[str, Any], model: DecisionModel = CLEF
) -> list[dict]:
    calls: list[dict[str, Any]] = []

    def post(url: str, token: str, body: dict[str, Any]) -> dict[str, Any]:
        calls.append({"url": url, "token": token, "body": body})
        req = _request(body)
        answer = by_title[req["state"]["this_step"]["title"]]
        if isinstance(answer, list):  # one answer per phrasing, v0..v2
            criteria = req["questions"]["kind"]["criteria"]
            variant = next(v for v, c in dm.KINDS.items() if c == criteria)
            answer = answer[int(variant[1])]
        return _as_served(model, answer)

    monkeypatch.setattr(dm, "_post", post)
    return calls


@MODELS
def test_the_request_is_the_pre_registered_one(
    model: DecisionModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, creds: None
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
        model,
    )
    work = tmp_path / "work"
    work.mkdir()
    system1.prepare(_skill(tmp_path), work, decider=model.name)
    assert len(calls) == 9  # 3 steps x 3 phrasings
    first = calls[0]
    assert first["token"] == "tok"
    body = first["body"]
    base = "https://api.cloudflare.com/client/v4/accounts/acct123/ai/run"
    if model is CLEF:
        assert first["url"] == f"{base}/@cf/cloudflare/clef"
        assert set(body) == {"model", "state", "questions"} and body["model"] == "clef"
    else:
        # The unified route: model id in the body, the request under `input`.
        assert first["url"] == base
        assert set(body) == {"model", "input"} and body["model"] == "typesafe/jev-1.14.0"
    body = _request(body)
    assert set(body["state"]) == {"skill", "skill_description", "all_steps", "this_step"}
    assert body["state"]["all_steps"] == [
        "Step 1: Total the batch",
        "Step 2: Tidy the notes",
        "Step 3: Score the leads",
    ]
    # The calls run in parallel, so group them by step instead of trusting order.
    step1 = [
        _request(c["body"])["questions"]
        for c in calls
        if _request(c["body"])["state"]["this_step"]["title"] == "Step 1: Total the batch"
    ]
    pairs = sorted((q["kind"]["criteria"] == dm.KINDS[v], v) for q in step1 for v in dm.KINDS)
    assert sorted(v for ok, v in pairs if ok) == ["v0", "v1", "v2"]
    assert sorted(q["human"]["instructions"] for q in step1) == sorted(dm.HUMAN.values())
    for q in step1:
        variant = next(v for v, c in dm.KINDS.items() if c == q["kind"]["criteria"])
        assert q["human"]["instructions"] == dm.HUMAN[variant]


def test_the_calls_run_in_parallel(monkeypatch: pytest.MonkeyPatch, creds: None) -> None:
    """Nine independent calls must not take nine call latencies: the first two
    only return once both are in flight, which a serial loop never reaches."""
    import tempfile
    import threading

    from rote.compiler.system1.parse import parse_skill

    both_in_flight = threading.Barrier(2, timeout=5)
    arrivals = {"n": 0}
    lock = threading.Lock()

    def meet(url: str, token: str, body: dict[str, Any]) -> dict[str, Any]:
        with lock:
            arrivals["n"] += 1
            first_two = arrivals["n"] <= 2
        if first_two:
            both_in_flight.wait()  # BrokenBarrierError when the calls are serial
        return _answers({**ZERO, "pure_function": 1.0}, 0.0)

    monkeypatch.setattr(dm, "_post", meet)
    with tempfile.TemporaryDirectory() as t:
        verdicts = dm.decide(CLEF, parse_skill(_skill(Path(t))))
    assert set(verdicts.by_unit) == {0, 1, 2}


def test_the_thresholds_are_the_frozen_ones() -> None:
    """Fitted on the dev skills, then measured once on the holdout: an edit
    here invalidates docs/compile-bench.md's numbers."""
    assert (CLEF.kind_threshold, CLEF.human_threshold) == (0.85, 0.50)
    assert (JEV.kind_threshold, JEV.human_threshold) == (JEV_KIND, JEV_HUMAN)
    assert (CLEF.price_per_m_input, JEV.price_per_m_input) == (0.24, 0.042)
    assert set(dm.MODELS) == {"clef", "jev"} and system1.DECIDERS == ("rules", "clef", "jev")


def _just_above(x: float) -> float:
    return min(round(x + 0.02, 2), 1.0)


def _just_below(x: float) -> float:
    return round(x - 0.02, 2)


@MODELS
def test_confident_answers_win_and_unsure_ones_go_to_review(
    model: DecisionModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, creds: None
) -> None:
    """Each answer sits just across one of ``model``'s own thresholds, so the
    other model's thresholds would route at least one step differently."""
    kind_hi, kind_lo = _just_above(model.kind_threshold), _just_below(model.kind_threshold)
    human_hi, human_lo = _just_above(model.human_threshold), _just_below(model.human_threshold)
    _stub(
        monkeypatch,
        {
            # The rules say pure_function; the model is sure a person must act.
            "Step 1: Total the batch": _answers({**ZERO, "pure_function": 0.9}, human_hi),
            # No signal for the rules (pure_function, 0.3); the model is sure it is a call.
            "Step 2: Tidy the notes": _answers({**ZERO, "external_call": kind_hi}, human_lo),
            # The rules say llm_judge. The model's first phrasing is sure it is
            # pure_function, but the three-phrasing mean is just below the threshold.
            "Step 3: Score the leads": [
                _answers({**ZERO, "pure_function": 1.0}, 0.0),
                _answers({**ZERO, "pure_function": 1.5 * kind_lo - 0.5}, 0.0),
                _answers({**ZERO, "pure_function": 1.5 * kind_lo - 0.5}, 0.0),
            ],
        },
        model,
    )
    work = tmp_path / "work"
    work.mkdir()
    plan = system1.prepare(_skill(tmp_path), work, decider=model.name)
    doc = yaml.safe_load((work / "pipeline.yaml").read_text())
    kinds = {n["source"]["section"]: n["kind"] for n in doc["nodes"]}
    assert kinds["Step 1: Total the batch"] == "hitl_gate"
    assert kinds["Step 2: Tidy the notes"] == "external_call"
    assert kinds["Step 3: Score the leads"] == "llm_judge"  # the rules decide below threshold
    assert any(f"{model.label} disagrees below its threshold" in r for r in plan.review)
    decisions = json.loads((work / "system1" / "decisions.json").read_text())["decider"]
    assert decisions["name"] == model.name and decisions["model"] == model.route
    # 9 calls x 1000 input tokens at the model's own price.
    assert decisions["cost_usd"] == round(9 * 1000 / 1e6 * model.price_per_m_input, 6)
    report = system1.report_extras(work, plan).preamble
    assert f"rules and the {model.label} decision model" in report
    spent = round(9 * 1000 / 1e6 * model.price_per_m_input, 4)
    assert f"Its model calls cost ${spent:.4f} ({model.label})" in report


@MODELS
@pytest.mark.asyncio
async def test_asking_for_a_model_without_credentials_fails_the_compile(
    model: DecisionModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
    monkeypatch.delenv("CLOUDFLARE_API_TOKEN", raising=False)

    def no_call(url: str, token: str, body: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError("the model was called without credentials")

    monkeypatch.setattr(dm, "_post", no_call)
    monkeypatch.setattr(Compiler, "select_driver", lambda self: _NeverRuns())
    compiler = Compiler(
        system1_decider=model.name, system1_tiebreak="none", system1_structure="rules"
    )
    with pytest.raises(
        CompilerError, match=f"--decider {model.name} needs CLOUDFLARE_ACCOUNT_ID and"
    ):
        await compiler.compile(_skill(tmp_path), tmp_path / "out")


def test_a_rate_limit_is_retried_and_a_dead_endpoint_is_loud(
    monkeypatch: pytest.MonkeyPatch, creds: None
) -> None:
    attempts = {"n": 0}

    def flaky(url: str, token: str, body: dict[str, Any]) -> dict[str, Any]:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise dm.DeciderUnavailable("Workers AI answered HTTP 429")
        return _answers({**ZERO, "pure_function": 1.0}, 0.0)

    monkeypatch.setattr(dm, "_post", flaky)
    assert dm.ask(CLEF, "a", "t", {}, "v0")["answers"]["human"]["noul"] == 0.0
    assert attempts["n"] == 3

    def dead(url: str, token: str, body: dict[str, Any]) -> dict[str, Any]:
        raise dm.DeciderUnavailable("Workers AI answered HTTP 500")

    monkeypatch.setattr(dm, "_post", dead)
    with pytest.raises(dm.DeciderUnavailable, match="did not answer after 6 tries"):
        dm.ask(CLEF, "a", "t", {}, "v0")


def test_a_refused_request_fails_at_once(monkeypatch: pytest.MonkeyPatch, creds: None) -> None:
    """A 401/403/404 fails the same way every time: no six rounds of backoff."""
    attempts = {"n": 0}

    def refused(url: str, token: str, body: dict[str, Any]) -> dict[str, Any]:
        attempts["n"] += 1
        raise dm.DeciderRejected("Workers AI answered HTTP 401: b'bad token'")

    monkeypatch.setattr(dm, "_post", refused)
    with pytest.raises(dm.DeciderRejected, match="HTTP 401"):
        dm.ask(CLEF, "a", "t", {}, "v0")
    assert attempts["n"] == 1


def test_an_empty_gateway_balance_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    import io
    import urllib.error

    def raise_402(req: object, timeout: float) -> object:
        raise urllib.error.HTTPError(
            "u", 402, "m", {}, io.BytesIO(b'{"errors":[{"message":"Insufficient balance"}]}')
        )  # type: ignore[arg-type]

    monkeypatch.setattr(dm.urllib.request, "urlopen", raise_402)
    with pytest.raises(dm.DeciderRejected, match="HTTP 402.*Unified Billing credits"):
        dm._post("https://example.invalid", "t", {})


@pytest.mark.parametrize(
    ("code", "rejected"),
    [(401, True), (402, True), (403, True), (404, True), (429, False), (500, False)],
)
def test_http_status_maps_to_retry_or_not(
    monkeypatch: pytest.MonkeyPatch, code: int, rejected: bool
) -> None:
    import io
    import urllib.error

    def raise_http(req: object, timeout: float) -> object:
        raise urllib.error.HTTPError("u", code, "m", {}, io.BytesIO(b"body"))  # type: ignore[arg-type]

    monkeypatch.setattr(dm.urllib.request, "urlopen", raise_http)
    with pytest.raises(dm.DeciderUnavailable) as err:
        dm._post("https://example.invalid", "t", {})
    assert isinstance(err.value, dm.DeciderRejected) is rejected


def test_a_decider_flag_needs_system1(tmp_path: Path) -> None:
    from rote import cli

    code = cli.main(
        [
            "compile",
            str(_skill(tmp_path)),
            "--out",
            str(tmp_path / "o"),
            "--no-system1",
            "--decider",
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

    monkeypatch.setattr(dm.urllib.request, "urlopen", capture)
    dm._post("https://example.invalid", "t", {})
    return sent


def test_gateway_metadata_tags_every_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """A host meters a decider per compile by filtering gateway logs on this metadata."""
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
@MODELS
def test_bad_gateway_metadata_fails_before_any_call(
    model: DecisionModel,
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
    with pytest.raises(dm.DeciderUnavailable, match=match):
        system1.prepare(_skill(tmp_path), work, decider=model.name)
    assert calls == []


def test_five_scalar_entries_are_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    meta = {"a": "s", "b": 1, "c": 1.5, "d": True, "graduation_id": "g1"}
    monkeypatch.setenv("CLOUDFLARE_AI_GATEWAY_ID", "rote")
    monkeypatch.setenv("CLOUDFLARE_AI_GATEWAY_METADATA", json.dumps(meta))
    assert json.loads(_sent_headers(monkeypatch)["cf-aig-metadata"]) == meta
