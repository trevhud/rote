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

Credentials are the user's own Cloudflare account, from the environment
(``CLOUDFLARE_ACCOUNT_ID``, ``CLOUDFLARE_API_TOKEN``). Missing credentials
fail the compile. There is no silent fallback to rules: asking for a model
and getting rules would report numbers for a decider that never ran.
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
)

#: Every model decider, by ``--decider`` value.
MODELS: dict[str, DecisionModel] = {m.name: m for m in (CLEF, JEV)}

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


def credentials(model: DecisionModel) -> tuple[str, str]:
    account = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "").strip()
    token = os.environ.get("CLOUDFLARE_API_TOKEN", "").strip()
    if not account or not token:
        raise DeciderUnavailable(
            f"--decider {model.name} needs CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN "
            "for the Cloudflare account that runs Workers AI (use a personal account, not "
            "an ambient work login)"
        )
    return account, token


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
        if value is None or not isinstance(value, str | int | float | bool):
            raise DeciderUnavailable(
                f"CLOUDFLARE_AI_GATEWAY_METADATA key {key!r}: values must be a string, "
                "number or boolean"
            )
    return parsed


def _post(url: str, token: str, body: dict[str, Any]) -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        **_gateway_headers(),
    }
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers=headers, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            loaded = json.loads(r.read())
    except urllib.error.HTTPError as e:
        detail = f"Workers AI answered HTTP {e.code}: {e.read()[:300]!r}"
        if 400 <= e.code < 500 and e.code != 429:
            # A bad token, a missing model or a malformed body fails the same
            # way every time: fail now instead of six times with backoff.
            raise DeciderRejected(detail) from None
        raise DeciderUnavailable(detail) from None
    return loaded if isinstance(loaded, dict) else {}


def _answers(resp: dict[str, Any]) -> dict[str, Any] | None:
    """The ``{"answers", "usage"}`` record, wherever the route put it.

    A per-model route returns it as ``result``. The unified ``/ai/run``
    route wraps it once more, as ``result.result``."""
    res = resp.get("result")
    if isinstance(res, dict) and "answers" not in res and isinstance(res.get("result"), dict):
        res = res["result"]
    return res if isinstance(res, dict) and "answers" in res else None


def ask(
    model: DecisionModel, account: str, token: str, state: dict[str, Any], variant: str
) -> dict[str, Any]:
    """One call: {"answers": ..., "usage": ...}. Retries 429s and transient errors."""
    url = model.url(account)
    body = model.body(state, questions_for(variant))
    last: Exception | None = None
    for attempt in range(6):
        try:
            resp = _post(url, token, body)
        except DeciderRejected:
            raise
        except (DeciderUnavailable, OSError) as e:
            last = e
            time.sleep(min(4 * (attempt + 1), 20))
            continue
        res = _answers(resp)
        if res is not None:
            return res
        last = DeciderUnavailable(f"Workers AI returned no answers: {json.dumps(resp)[:300]}")
        time.sleep(min(4 * (attempt + 1), 20))
    raise DeciderUnavailable(f"{model.label} did not answer after 6 tries: {last}")


def decide(model: DecisionModel, skill: SkillParse) -> Verdicts:
    """The model's ensembled kind and gate probability for every step unit."""
    account, token = credentials(model)
    _gateway_headers()  # a bad gateway setting fails here, not six retries later
    out: dict[int, Verdict] = {}
    for u in skill.units:
        state = state_for(skill, u)
        probs: dict[str, list[float]] = {}
        human: list[float] = []
        cost = 0.0
        for variant in ("v0", "v1", "v2"):
            res = ask(model, account, token, state, variant)
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
