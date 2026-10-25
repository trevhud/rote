"""``rote mcp login --no-wait`` as Actions, for the command and for ``rote setup``.

A login an agent can hand to its user (Stripe's pending-login shape,
:mod:`rote.mcp.pending_login`): start a detached listener that waits for
the browser, optionally open the authorization page, and return. The two
Actions below are the whole side effect; ``rote mcp login --no-wait`` and
``rote setup``'s login step (agent mode) both apply them, so the plan an
agent relays is the one that runs.
"""

from __future__ import annotations

import webbrowser
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from rote.interact.plan import Action, Contact, FileWrite, Launch

if TYPE_CHECKING:
    from rote.mcp.pending_login import Started
    from rote.mcp.registry import McpServerConfig


def no_wait_actions(
    server: str, config: McpServerConfig, *, browser: bool = True
) -> tuple[list[Action], list[Started]]:
    """The actions that start ``server``'s login without waiting, and the
    list the start action appends its :class:`~rote.mcp.pending_login.Started`
    to when it runs (empty until then)."""
    from rote.mcp.auth import CALLBACK_PORTS
    from rote.mcp.pending_login import PENDING_TTL_SECONDS, pending_dir, start_pending_login

    host = urlsplit(config.url).hostname or config.url
    minutes = PENDING_TTL_SECONDS // 60
    ports = f"{CALLBACK_PORTS[0]}-{CALLBACK_PORTS[-1]}"
    started: list[Started] = []

    def start() -> str:
        started.append(start_pending_login(server, config))
        return started[0].kind

    actions = [
        Action(
            id=f"start-login-{server}",
            title=f"Log in to {server}: a browser page for you to approve, waited on for up "
            f"to {minutes} min",
            effects=(
                Contact(
                    host=host,
                    purpose="find its authorization server, and register rote as a "
                    "client there if it needs one",
                ),
                Launch(
                    program=f"a rote callback listener on localhost (one of ports {ports})",
                    purpose=f"receive the browser's return, for at most {minutes} minutes",
                ),
                FileWrite(
                    path=pending_dir() / "<id>.json",
                    summary="the pending login record, owner-only",
                    file_mode=0o600,
                ),
            ),
            run=start,
            command=f"rote mcp login {server} --no-wait",
        )
    ]
    if browser:

        def open_browser() -> bool:
            if not started or started[0].authorization_url is None:
                return False
            return bool(webbrowser.open(started[0].authorization_url))

        actions.append(
            Action(
                id=f"open-browser-{server}",
                title=f"Open {server}'s authorization page",
                effects=(Launch(program="your web browser", purpose="approve the login"),),
                run=open_browser,
                command=f"rote mcp login {server} --no-wait --no-browser",
            )
        )
    return actions, started
