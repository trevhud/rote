"""Cloudflare Clef as a System-1 decider (EXPERIMENTAL, ``--system1-decider clef``).

Clef is a decision model on Workers AI: a state plus typed questions in,
calibrated probabilities out, input-priced, no generated text. It answers
two questions per step, kind and "does a person have to act here", where
the `models` lane measured it beating the rules on held-out skills (kind
choice 0.96 against the rules' 0.65 Jaccard, gates 3 of 3 against 1 of 3).
Mandatory and fan_out stay on rules, which won there.

Everything below is fixed as that lane pre-registered it on the five dev
skills, before the held-out run: the three phrasings, the state, the
ensemble (mean of the per-option probabilities over the phrasings), and the
thresholds. Changing any of them invalidates the measurement.

Credentials are the user's own Cloudflare account, from the environment
(``CLOUDFLARE_ACCOUNT_ID``, ``CLOUDFLARE_API_TOKEN``). Missing credentials
fail the compile. There is no silent fallback to rules: asking for clef and
getting rules would report numbers for a decider that never ran.
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

MODEL_ROUTE = "@cf/cloudflare/clef"
MODEL_SELECTOR = "clef"
#: $ per million input tokens; output is free.
PRICE_PER_M_INPUT = 0.24
#: Above this ensemble confidence the model's kind wins (14 of 14 right on held-out).
KIND_THRESHOLD = 0.85
#: The "a person must act" probability at or above which a step is a gate,
#: fitted on the dev skills (every held-out gate was above 0.8, every other
#: step below 0.2).
HUMAN_THRESHOLD = 0.50

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


class ClefUnavailable(RuntimeError):
    """--system1-decider clef was asked for and cannot run."""


class ClefRejected(ClefUnavailable):
    """Workers AI refused the request (a 4xx other than 429): retrying cannot help."""


@dataclass(frozen=True)
class Verdict:
    kind: str
    kind_confidence: float
    human: float
    cost_usd: float


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


def credentials() -> tuple[str, str]:
    account = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "").strip()
    token = os.environ.get("CLOUDFLARE_API_TOKEN", "").strip()
    if not account or not token:
        raise ClefUnavailable(
            "--system1-decider clef needs CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN "
            "for the Cloudflare account that runs Workers AI (use a personal account, not "
            "an ambient work login)"
        )
    return account, token


def _gateway_headers() -> dict[str, str]:
    """Route through a named AI Gateway, tagging each logged call.

    ``CLOUDFLARE_AI_GATEWAY_METADATA`` is a JSON object the gateway stores on
    every log, which is how a host attributes Clef's cost to one compile.
    Metadata with no gateway would be dropped by Workers AI, so it is an error.
    """
    headers: dict[str, str] = {}
    gateway = os.environ.get("CLOUDFLARE_AI_GATEWAY_ID", "").strip()
    metadata = os.environ.get("CLOUDFLARE_AI_GATEWAY_METADATA", "").strip()
    if gateway:
        headers["cf-aig-gateway-id"] = gateway
    if metadata:
        if not gateway:
            raise ClefUnavailable(
                "CLOUDFLARE_AI_GATEWAY_METADATA is set but CLOUDFLARE_AI_GATEWAY_ID is not: "
                "only a gateway records metadata"
            )
        try:
            parsed = json.loads(metadata)
        except json.JSONDecodeError as e:
            raise ClefUnavailable(f"CLOUDFLARE_AI_GATEWAY_METADATA is not JSON: {e}") from None
        if not isinstance(parsed, dict):
            raise ClefUnavailable("CLOUDFLARE_AI_GATEWAY_METADATA must be a JSON object")
        headers["cf-aig-metadata"] = json.dumps(parsed)
    return headers


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
            raise ClefRejected(detail) from None
        raise ClefUnavailable(detail) from None
    return loaded if isinstance(loaded, dict) else {}


def ask(account: str, token: str, state: dict[str, Any], variant: str) -> dict[str, Any]:
    """One call: {"answers": ..., "usage": ...}. Retries 429s and transient errors."""
    url = f"https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/{MODEL_ROUTE}"
    body = {"model": MODEL_SELECTOR, "state": state, "questions": questions_for(variant)}
    last: Exception | None = None
    for attempt in range(6):
        try:
            resp = _post(url, token, body)
        except ClefRejected:
            raise
        except (ClefUnavailable, OSError) as e:
            last = e
            time.sleep(min(4 * (attempt + 1), 20))
            continue
        res = resp.get("result") or {}
        if isinstance(res, dict) and "answers" in res:
            return res
        last = ClefUnavailable(f"Workers AI returned no answers: {json.dumps(resp)[:300]}")
        time.sleep(min(4 * (attempt + 1), 20))
    raise ClefUnavailable(f"Clef did not answer after 6 tries: {last}")


def decide(skill: SkillParse) -> dict[int, Verdict]:
    """Clef's ensembled kind and gate probability for every step unit."""
    account, token = credentials()
    _gateway_headers()  # a bad gateway setting fails here, not six retries later
    out: dict[int, Verdict] = {}
    for u in skill.units:
        state = state_for(skill, u)
        probs: dict[str, list[float]] = {}
        human: list[float] = []
        cost = 0.0
        for variant in ("v0", "v1", "v2"):
            res = ask(account, token, state, variant)
            answers = res["answers"]
            for k, p in answers["kind"]["probabilities"].items():
                probs.setdefault(k, []).append(float(p))
            human.append(float(answers["human"]["noul"]))
            usage = res.get("usage") or {}
            cost += float(usage.get("input_tokens", 0)) / 1e6 * PRICE_PER_M_INPUT
        mean = {k: sum(v) / len(v) for k, v in probs.items()}
        kind = max(mean, key=lambda k: mean[k])
        out[u.index] = Verdict(kind, round(mean[kind], 4), round(sum(human) / len(human), 4), cost)
    return out
