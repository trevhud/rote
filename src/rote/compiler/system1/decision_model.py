"""Decision models as System-1 deciders (``--decider clef`` / ``--decider jev``).

A decision model on Workers AI takes a state plus typed questions and
returns calibrated probabilities, input-priced, with no generated text. Two
are wired in, behind one client:

- Cloudflare's Clef (``@cf/cloudflare/clef``).
- Typesafe's Jev (``typesafe/jev-1.14.0``), the same state and typed-question
  API family, served on the unified ``/ai/run`` route.

Each answers two questions per step: its kind, and "does a person have to
act here". Mandatory and fan_out stay on rules, which won on both.

The phrasings, the state and the ensemble (the mean of the per-option
probabilities over the three phrasings) are shared and were pre-registered
by the `models` lane on the five dev skills, before any held-out run. Each
model's two thresholds were fitted on the dev skills only and then frozen
before the one measurement on ``examples/holdout/``. Changing any of them
invalidates that measurement. The numbers are in ``docs/compile-bench.md``.

Clef runs on Workers AI with ``ROTE_CLEF_ACCOUNT_ID`` and
``ROTE_CLEF_API_TOKEN``, and Jev on Typesafe's own API with
``TYPESAFE_API_KEY`` (:func:`route`). Never ``CLOUDFLARE_*``: those are the
credentials ``rote deploy`` and ``rote mcp release`` use, and having them must
not opt a compile into sending skill text to an account that gets billed for
it (on a work machine, often the employer's). An explicit model with no
credentials for it fails the
compile: asking for a model and getting rules would report numbers for a
decider that never ran. The built-in default (``auto``) instead takes the
first of :data:`CHECKER_ORDER` that has credentials, and none is no check
(:func:`default_checker`).
``CLOUDFLARE_AI_GATEWAY_ID`` routes the calls through that AI Gateway, and
``CLOUDFLARE_AI_GATEWAY_METADATA`` (a JSON object) tags each logged call.

Stdlib only (urllib), so it adds no dependency.
"""

# The phrasings below are verbatim pre-registered text; wrapping them would
# change nothing, and splitting them invites an edit that changes the measure.
# ruff: noqa: E501

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from rote.compiler.system1.parse import SkillParse, Unit


@dataclass(frozen=True)
class DecisionModel:
    """One Workers AI decision model and its frozen thresholds."""

    #: The ``--decider`` value.
    name: str
    #: How the report names it.
    label: str
    #: The Workers AI model id.
    route: str
    #: The ``model`` field of a per-model route's body. ``None``: the model
    #: is served on the unified ``/ai/run`` route, which takes the model id
    #: in the body and the request under ``input``.
    selector: str | None
    #: $ per million input tokens; output is free.
    price_per_m_input: float
    #: At or above this ensemble confidence, the model's kind wins.
    kind_threshold: float
    #: The "a person must act" probability at or above which a step is a gate.
    human_threshold: float
    #: The most input tokens a call takes. Workers AI truncates a longer text
    #: state to fit, without an error, so a state that might not fit is
    #: refused instead (:func:`decide`).
    context_tokens: int

    def url(self, account: str) -> str:
        base = f"https://api.cloudflare.com/client/v4/accounts/{account}/ai/run"
        return base if self.selector is None else f"{base}/{self.route}"

    def body(self, state: dict[str, Any], questions: dict[str, Any]) -> dict[str, Any]:
        if self.selector is None:
            return {"model": self.route, "input": {"state": state, "questions": questions}}
        return {"model": self.selector, "state": state, "questions": questions}


CLEF = DecisionModel(
    name="clef",
    label="Clef",
    route="@cf/cloudflare/clef",
    selector="clef",
    price_per_m_input=0.24,
    # 14 of 14 kinds right on held-out at or above it.
    kind_threshold=0.85,
    # Every held-out gate was above 0.8, every other step below 0.2.
    human_threshold=0.50,
    # Documented: "65,536 tokens" context, and a longer text state "is
    # truncated to fit" (developers.cloudflare.com/workers-ai/models/clef/).
    context_tokens=65_536,
)

JEV = DecisionModel(
    name="jev",
    label="Jev",
    route="typesafe/jev-1.14.0",
    selector=None,
    price_per_m_input=0.042,
    # Fitted on the dev skills only: the lowest value with its kind choice
    # right on 95% of the steps at or above it (21 of 22 at 0.85).
    kind_threshold=0.85,
    # The best dev gate F1 (4 of 4 gates, 1 false), the higher value of a tie.
    human_threshold=0.60,
    # ASSUMPTION: Cloudflare publishes no model page for Jev. Third-party
    # listings disagree (cloudprice.net/models/typesafe-ai-jev-1-14: 32K;
    # requesty.ai/models/typesafe/jev-1.14.0: 64K), so the smaller one, which
    # refuses a state that might be truncated rather than risking it.
    context_tokens=32_768,
)

#: Every model decider, by ``--decider`` value.
MODELS: dict[str, DecisionModel] = {m.name: m for m in (CLEF, JEV)}

#: The built-in default checks with the first of these that has credentials,
#: best measured first. Set by sk-check2 (2026-10-21, the structure pass with
#: each as its checker, 2 trials on the dev and held-out skills, numbers in
#: docs/design/system1-compile.md): Clef made the one change of the round, and
#: it was right (customer-onboarding's sanctions screen is an external_call),
#: while Jev made none. A later bench changes the order here and nowhere else.
CHECKER_ORDER: tuple[DecisionModel, ...] = (CLEF, JEV)


def default_checker() -> tuple[str, str]:
    """``(decider, why)`` for the built-in ``auto``: the first of
    :data:`CHECKER_ORDER` with credentials, else ``rules`` (no check)."""
    for model in CHECKER_ORDER:
        try:
            via = route(model)
        except DeciderUnavailable:
            continue
        return model.name, (
            f"the default: {model.label}, the best-measured decision model with credentials "
            f"here ({via.via})"
        )
    return "rules", (
        f"no decision model ran (no credentials for one): set {CLEF_ACCOUNT_ENV} and "
        f"{CLEF_TOKEN_ENV} for Clef, or {TYPESAFE_KEY_ENV} for Jev"
    )


KINDS = {
    "v0": {
        "pure_function": "Fixed logic code can do exactly: filter, match, compute, format, apply stated rules",
        "external_call": "Reads from or writes to an outside system: an API, tool, website, inbox or file store",
        "llm_judge": "Needs judgment over natural language: classify, score with a rubric, draft or interpret text",
        "agent_loop": "Open-ended exploration whose number of actions depends on what is found",
        "hitl_gate": "Stops and waits for a person to provide input or approve",
    },
    "v1": {
        "pure_function": "deterministic computation",
        "external_call": "call to an external system",
        "llm_judge": "language-model judgment",
        "agent_loop": "autonomous multi-step exploration",
        "hitl_gate": "human approval or input",
    },
    "v2": {
        "pure_function": "Same input always gives the same output; no outside system and no judgment",
        "external_call": "One request to an outside system whose result the workflow uses or whose side effect it needs",
        "llm_judge": "A bounded decision over text with a fixed set of possible answers, made by a language model",
        "agent_loop": "A bounded loop where a model chooses which tools to call next until a goal is met",
        "hitl_gate": "The workflow pauses durably until a human sends a decision or data",
    },
}
HUMAN = {
    "v0": "Does this step stop and wait for a person to provide input or approval before the workflow continues?",
    "v1": "Does this step wait for a human?",
    "v2": "Must a human act or answer before the steps after this one can run?",
}
KIND_INSTRUCTIONS = "Which kind of workflow node best describes the main work of this step?"


class DeciderUnavailable(RuntimeError):
    """A model decider was asked for and cannot run."""


class DeciderRejected(DeciderUnavailable):
    """Workers AI refused the request (a 4xx other than 429): retrying cannot help."""


@dataclass(frozen=True)
class Verdict:
    kind: str
    kind_confidence: float
    human: float
    cost_usd: float


@dataclass(frozen=True)
class Verdicts:
    """One model's verdict on every step unit, by unit index."""

    model: DecisionModel
    by_unit: dict[int, Verdict]

    @property
    def cost_usd(self) -> float:
        return sum(v.cost_usd for v in self.by_unit.values())


def state_for(skill: SkillParse, u: Unit) -> dict[str, Any]:
    return {
        "skill": skill.name,
        "skill_description": skill.description,
        "all_steps": [x.title for x in skill.units],
        "this_step": {"section": u.section, "title": u.title, "text": u.text},
    }


def questions_for(variant: str) -> dict[str, Any]:
    return {
        "kind": {"type": "choice", "instructions": KIND_INSTRUCTIONS, "criteria": KINDS[variant]},
        "human": {"type": "noul", "instructions": HUMAN[variant]},
    }


#: Typesafe's own endpoint for Jev (docs.typesafe.ai/api: ``POST /v1/systemone``,
#: ``Authorization: Bearer <API_KEY>``; the body is ``model``, ``state``,
#: ``questions``, and the answer ``{model, answers, usage}`` with no wrapper).
TYPESAFE_URL = "https://api.typesafe.ai/v1/systemone"
#: Where Typesafe's key is read from. Typesafe's docs name no variable; this is
#: the one LiteLLM's Typesafe pass-through uses.
TYPESAFE_KEY_ENV = "TYPESAFE_API_KEY"
#: The Jev version its frozen thresholds were fitted on, pinned rather than
#: ``jev-latest`` so a new release cannot move the measurement.
TYPESAFE_JEV_MODEL = "jev-1.14.0"


@dataclass(frozen=True)
class Route:
    """Where one model's calls go, and with which key."""

    url: str
    token: str
    body: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]
    #: Who answers, for messages and the report.
    via: str
    #: Workers AI calls may go through an AI Gateway; Typesafe's never do.
    cloudflare: bool


#: Clef's credentials: a Cloudflare account and an API token that can run
#: Workers AI, set for rote's decision model alone.
CLEF_ACCOUNT_ENV = "ROTE_CLEF_ACCOUNT_ID"
CLEF_TOKEN_ENV = "ROTE_CLEF_API_TOKEN"


def clef_credentials() -> tuple[str, str] | None:
    account = os.environ.get(CLEF_ACCOUNT_ENV, "").strip()
    token = os.environ.get(CLEF_TOKEN_ENV, "").strip()
    return (account, token) if account and token else None


def typesafe_key() -> str | None:
    return os.environ.get(TYPESAFE_KEY_ENV, "").strip() or None


def route(model: DecisionModel) -> Route:
    """The model's calls: Clef on Workers AI, Jev on Typesafe's own API, each
    with its own rote variables. No credentials for the model fails, naming
    what to set."""
    if model.name == JEV.name:
        key = typesafe_key()
        if key is None:
            raise DeciderUnavailable(
                f"--decider jev needs {TYPESAFE_KEY_ENV}, a key for Typesafe's API "
                "(console.typesafe.ai)"
            )
        return Route(
            TYPESAFE_URL,
            key,
            lambda state, questions: {
                "model": TYPESAFE_JEV_MODEL,
                "state": state,
                "questions": questions,
            },
            "Typesafe",
            cloudflare=False,
        )
    clef = clef_credentials()
    if clef is None:
        raise DeciderUnavailable(
            f"--decider {model.name} needs {CLEF_ACCOUNT_ENV} and {CLEF_TOKEN_ENV}: a "
            "Cloudflare account and an API token that can run Workers AI, set for rote's "
            "decision model alone (CLOUDFLARE_* deploy credentials are never used for it)"
        )
    account, token = clef
    return Route(model.url(account), token, model.body, "Workers AI", cloudflare=True)


def _gateway_headers() -> dict[str, str]:
    """Route through a named AI Gateway, tagging each logged call.

    ``CLOUDFLARE_AI_GATEWAY_METADATA`` is a JSON object the gateway stores on
    every log, which is how a host attributes a decider's cost to one compile.
    Metadata with no gateway would be dropped by Workers AI, so it is an error.
    """
    headers: dict[str, str] = {}
    gateway = os.environ.get("CLOUDFLARE_AI_GATEWAY_ID", "").strip()
    metadata = os.environ.get("CLOUDFLARE_AI_GATEWAY_METADATA", "").strip()
    if gateway:
        headers["cf-aig-gateway-id"] = gateway
    if metadata:
        if not gateway:
            raise DeciderUnavailable(
                "CLOUDFLARE_AI_GATEWAY_METADATA is set but CLOUDFLARE_AI_GATEWAY_ID is not: "
                "only a gateway records metadata"
            )
        try:
            parsed = json.loads(metadata)
        except json.JSONDecodeError as e:
            raise DeciderUnavailable(f"CLOUDFLARE_AI_GATEWAY_METADATA is not JSON: {e}") from None
        headers["cf-aig-metadata"] = json.dumps(_checked_metadata(parsed), separators=(",", ":"))
    return headers


#: AI Gateway keeps at most this many metadata entries and drops the rest
#: without an error (developers.cloudflare.com/ai-gateway/observability/custom-metadata/).
GATEWAY_METADATA_MAX_ENTRIES = 5


def _checked_metadata(parsed: object) -> dict[str, str | int | float | bool]:
    """Refuse metadata the gateway would silently change.

    The gateway drops entries past the fifth, strips keys starting with
    ``cf.``, and accepts only string, number and boolean values. Any of those
    could drop the key a host bills on, so each is an error here.
    """
    if not isinstance(parsed, dict):
        raise DeciderUnavailable("CLOUDFLARE_AI_GATEWAY_METADATA must be a JSON object")
    if len(parsed) > GATEWAY_METADATA_MAX_ENTRIES:
        raise DeciderUnavailable(
            f"CLOUDFLARE_AI_GATEWAY_METADATA has {len(parsed)} entries; the gateway keeps "
            f"only {GATEWAY_METADATA_MAX_ENTRIES} and drops the rest"
        )
    for key, value in parsed.items():
        if key.startswith("cf."):
            raise DeciderUnavailable(
                f"CLOUDFLARE_AI_GATEWAY_METADATA key {key!r}: the gateway strips cf.* keys"
            )
        if not isinstance(value, str | int | float | bool):
            raise DeciderUnavailable(
                f"CLOUDFLARE_AI_GATEWAY_METADATA key {key!r}: values must be a string, "
                "number or boolean"
            )
    return parsed


def _post(
    url: str, token: str, body: dict[str, Any], via: str = "Workers AI", cloudflare: bool = True
) -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        **(_gateway_headers() if cloudflare else {}),
    }
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers=headers, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            loaded = json.loads(r.read())
    except urllib.error.HTTPError as e:
        detail = f"{via} answered HTTP {e.code}: {e.read()[:300]!r}"
        if e.code == 402 and cloudflare:
            # Calls routed through an AI Gateway bill its Unified Billing
            # credits; an empty balance answers 402.
            detail += (
                ". The AI Gateway's balance is empty: add credits in the Cloudflare "
                "dashboard (AI > AI Gateway), or unset CLOUDFLARE_AI_GATEWAY_ID"
            )
        if 400 <= e.code < 500 and e.code != 429:
            # A bad token, a missing model or a malformed body fails the same
            # way every time: fail now instead of six times with backoff.
            raise DeciderRejected(detail) from None
        raise DeciderUnavailable(detail) from None
    return loaded if isinstance(loaded, dict) else {}


def _answers(resp: dict[str, Any]) -> dict[str, Any] | None:
    """The ``{"answers", "usage"}`` record, wherever the route put it.

    A per-model route returns it as ``result``. The unified ``/ai/run``
    route wraps it once more, as ``result.result``. Typesafe's own API
    returns it at the top level."""
    if "answers" in resp:
        return resp
    res = resp.get("result")
    if isinstance(res, dict) and "answers" not in res and isinstance(res.get("result"), dict):
        res = res["result"]
    return res if isinstance(res, dict) and "answers" in res else None


def ask(model: DecisionModel, via: Route, state: dict[str, Any], variant: str) -> dict[str, Any]:
    """One call: {"answers": ..., "usage": ...}. Retries 429s and transient errors."""
    url, token = via.url, via.token
    body = via.body(state, questions_for(variant))
    last: Exception | None = None
    for attempt in range(6):
        try:
            resp = _post(url, token, body, via.via, via.cloudflare)
        except DeciderRejected:
            raise
        except (DeciderUnavailable, OSError) as e:
            last = e
            time.sleep(min(4 * (attempt + 1), 20))
            continue
        res = _answers(resp)
        if res is not None:
            return res
        last = DeciderUnavailable(f"{via.via} returned no answers: {json.dumps(resp)[:300]}")
        time.sleep(min(4 * (attempt + 1), 20))
    raise DeciderUnavailable(f"{model.label} did not answer after 6 tries: {last}")


#: Calls in flight at once. Every (step, phrasing) call is independent, so a
#: compile's 30-odd calls take about one call's latency instead of 30.
MAX_PARALLEL = 16
VARIANTS = ("v0", "v1", "v2")


#: Characters per token assumed when sizing a state before sending it: low,
#: so the estimate errs toward refusing a state that might not fit. Prose runs
#: near 4, but markdown tables, base64, hashes and code tokenize far denser
#: (review6, on o200k: 2.5 counted only 0.62 of a table's real tokens), and
#: a step can be mostly a table. 1.5 still allows about 88k characters a step
#: on Clef and 44k on Jev.
_CHARS_PER_TOKEN = 1.5


def _check_fits(model: DecisionModel, states: list[dict[str, Any]]) -> None:
    """Refuse a state the model would truncate: Workers AI does so without an
    error, and the verdict would then be on part of the step."""
    longest = max((len(json.dumps(s)) for s in states), default=0)
    estimate = int(longest / _CHARS_PER_TOKEN)
    if estimate > model.context_tokens * 0.9:
        raise DeciderUnavailable(
            f"{model.label} reads at most {model.context_tokens} tokens and truncates a longer "
            f"state without an error; this skill's longest step state is about {estimate} "
            "tokens. Use --decider rules for this skill"
        )


def decide(model: DecisionModel, skill: SkillParse) -> Verdicts:
    """The model's ensembled kind and gate probability for every step unit.

    A state the model would truncate is refused before any call, and a call
    whose input reached the window is refused after it: both would be a
    verdict on text the model never saw."""
    from concurrent.futures import ThreadPoolExecutor

    via = route(model)
    if via.cloudflare:
        _gateway_headers()  # a bad gateway setting fails here, not six retries later
    states = {u.index: state_for(skill, u) for u in skill.units}
    _check_fits(model, list(states.values()))
    jobs = [(u, variant) for u in skill.units for variant in VARIANTS]
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as pool:
        results = list(pool.map(lambda j: ask(model, via, states[j[0].index], j[1]), jobs))
    filled = [
        r
        for r in results
        if (r.get("usage") or {}).get("input_tokens", 0) >= model.context_tokens * 0.98
    ]
    if filled:
        raise DeciderUnavailable(
            f"{model.label} read {filled[0]['usage']['input_tokens']} tokens, its whole window: "
            "the state was likely truncated. Use --decider rules for this skill"
        )
    by_unit: dict[int, list[dict[str, Any]]] = {}
    for (u, _), res in zip(jobs, results, strict=True):
        by_unit.setdefault(u.index, []).append(res)
    out: dict[int, Verdict] = {}
    for u in skill.units:
        probs: dict[str, list[float]] = {}
        human: list[float] = []
        cost = 0.0
        for res in by_unit[u.index]:
            answers = res["answers"]
            for k, p in answers["kind"]["probabilities"].items():
                probs.setdefault(k, []).append(float(p))
            human.append(float(answers["human"]["noul"]))
            usage = res.get("usage") or {}
            cost += float(usage.get("input_tokens", 0)) / 1e6 * model.price_per_m_input
        mean = {k: sum(v) / len(v) for k, v in probs.items()}
        kind = max(mean, key=lambda k: mean[k])
        out[u.index] = Verdict(kind, round(mean[kind], 4), round(sum(human) / len(human), 4), cost)
    return Verdicts(model, out)
