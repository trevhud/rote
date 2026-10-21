"""System-1's one call site: the codex transport, and a bounded retry on every lane.

Every structure-pass and tie-break call goes through :func:`structured_call`.
A ``codex-cli`` lane runs ``codex exec --output-schema`` here; every other
lane goes to the inference helper. A transient failure (a timeout, a rate
limit, a 5xx, a dropped connection) is retried :data:`ATTEMPTS` times in all
with backoff, on the same lane and model, and then the real error is raised.
That is a retry, not a fallback: no other model or provider is ever tried.
Found on GLM 5.2, where Workers AI answered BDR's structure call with a 408.

The codex transport:

The codex driver has no SDK client to lend, but ``codex exec`` sends an
``--output-schema`` file as the Responses API's ``text.format`` with
``strict: true`` (codex-rs ``create_text_param_for_request``; exec sessions
are always strict in 0.163.0), which is the same constrained decoding the
Anthropic and OpenAI lanes use. So a codex compile's structure pass and
tie-break run on codex itself, with the user's own codex login, model and
provider config, and never on another provider's key or binary.

What it does not give: a token cap (codex exec has none, openai/codex#55279)
or a way to drop the shell tool. The read-only sandbox, ``--ephemeral`` and
a scratch working directory keep it from writing anything. Only the last
``agent_message`` is the answer: the schema applies to every message of the
turn (openai/codex#19816).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Calls in all for one answer, and the wait before each retry (seconds).
ATTEMPTS = 3
BACKOFF_S = (2.0, 8.0)
#: An HTTP status a retry can cure: request timeout, conflict, too early,
#: rate limit, and any server error (529 is Anthropic's overloaded).
_TRANSIENT_STATUS = re.compile(r"(?:HTTP |API Error: |status[ :]+)(408|409|425|429|5\d\d)\b")
#: Transport failures with no status: the SDKs' connection and timeout
#: errors (by class name, as the helper reports them) and a reset socket.
_TRANSIENT_TRANSPORT = re.compile(
    r"\b(APIConnectionError|APITimeoutError|ECONNRESET|Connection reset|connection reset)\b"
)
#: Waits between attempts; tests replace it.
_sleep = time.sleep

#: Lane provider name for this transport.
PROVIDER = "codex-cli"
#: Seconds a call may take. Codex reasons before it answers, and nothing caps it.
TIMEOUT_S = 900


@dataclass(frozen=True)
class CodexCli:
    """The codex driver's binary, lent to System-1 as its lane's client."""

    executable: str = "codex"


def command(cli: CodexCli, model: str, schema: Path, last: Path) -> list[str]:
    """The ``codex exec`` argv for one schema-locked answer, prompt on stdin."""
    args = [
        cli.executable,
        "exec",
        "--ephemeral",
        "--skip-git-repo-check",
        "--sandbox",
        "read-only",
        "--color",
        "never",
        "-c",
        'web_search="disabled"',
        "--json",
        "--output-schema",
        str(schema),
        "--output-last-message",
        str(last),
    ]
    if model:
        args += ["--model", model]
    return [*args, "-"]


def parse_events(stdout: str) -> tuple[str | None, dict[str, Any] | None, str | None]:
    """``(last agent message, turn usage, failure)`` from ``--json`` output."""
    message = usage = failure = None
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = event.get("type")
        item = event.get("item") or {}
        if kind == "item.completed" and item.get("type") == "agent_message":
            message = item.get("text")
        elif kind == "turn.completed":
            usage = event.get("usage")
        elif kind in ("turn.failed", "error"):
            failure = json.dumps(event.get("error") or event)[:500]
    return message, usage, failure


def usage_row(usage: dict[str, Any] | None) -> dict[str, Any]:
    """Codex's turn usage in the helper's four buckets.

    Codex counts cached tokens inside ``input_tokens``, like OpenAI, so they
    are split out; reasoning tokens are already inside ``output_tokens``."""
    u = usage or {}
    total = u.get("input_tokens")
    cached = u.get("cached_input_tokens") or 0
    return {
        "input_tokens": None if total is None else total - cached,
        "output_tokens": u.get("output_tokens"),
        "cache_write_tokens": u.get("cache_write_input_tokens"),
        "cache_read_tokens": cached,
    }


def call(
    *,
    node_id: str,
    cli: CodexCli,
    model: str,
    prompt: str,
    output_schema: dict[str, Any],
) -> tuple[dict[str, Any], list[tuple[str, dict[str, Any]]]]:
    """One answer from ``codex exec``: ``(payload, usage rows)``, like ``structured_call``."""
    with tempfile.TemporaryDirectory(prefix="rote-system1-codex-") as tmp:
        schema = Path(tmp) / "schema.json"
        schema.write_text(json.dumps(output_schema), encoding="utf-8")
        last = Path(tmp) / "last.json"
        try:
            proc = subprocess.run(
                command(cli, model, schema, last),
                input=prompt,
                capture_output=True,
                text=True,
                cwd=tmp,
                env=os.environ.copy(),
                timeout=TIMEOUT_S,
            )
        except (OSError, subprocess.TimeoutExpired) as e:
            raise RuntimeError(f"codex exec for {node_id} did not finish: {e}") from None
        message, usage, failure = parse_events(proc.stdout)
        # An error event can be a retried hiccup inside a turn that still
        # answered; only a failed exit or no answer at all is a failure.
        if proc.returncode != 0 or (message is None and failure):
            tail = (failure or proc.stderr or proc.stdout)[-500:]
            raise RuntimeError(f"codex exec for {node_id} failed (exit {proc.returncode}): {tail}")
        text = (
            message if message is not None else (last.read_text("utf-8") if last.exists() else "")
        )
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        raise RuntimeError(
            f"codex exec for {node_id} answered with no JSON: {text[:200]!r}"
        ) from None
    if not isinstance(payload, dict):
        raise RuntimeError(f"codex exec for {node_id} answered with a non-object")
    return payload, [(model or "codex", usage_row(usage))]


def transient(error: BaseException) -> bool:
    """Whether a failed call is worth the same call again: a retryable HTTP
    status or a dropped connection, as the lane reported it."""
    text = str(error)
    return bool(_TRANSIENT_STATUS.search(text) or _TRANSIENT_TRANSPORT.search(text))


def structured_call(**kwargs: Any) -> tuple[dict[str, Any], list[tuple[str, dict[str, Any]]]]:
    """``structured_call`` for every System-1 lane, retried on a transient failure."""
    for attempt in range(ATTEMPTS):
        try:
            return _once(**kwargs)
        except RuntimeError as e:
            if attempt == ATTEMPTS - 1 or not transient(e):
                if attempt:
                    raise RuntimeError(f"{e} (after {attempt + 1} attempts)") from e
                raise
            _sleep(BACKOFF_S[min(attempt, len(BACKOFF_S) - 1)])
    raise AssertionError("unreachable")  # pragma: no cover


def _once(**kwargs: Any) -> tuple[dict[str, Any], list[tuple[str, dict[str, Any]]]]:
    """One call: codex here, the rest in the helper."""
    if kwargs.get("provider") == PROVIDER:
        return call(
            node_id=kwargs["node_id"],
            cli=kwargs["sdk_client"],
            model=kwargs["model"],
            prompt=kwargs["prompt"],
            output_schema=kwargs["output_schema"],
        )
    from rote.inference._runtime_helper import structured_call as helper_call

    return helper_call(**kwargs)
