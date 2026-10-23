"""Which AI client rote is running under, from the environment.

Two questions with different answers, so two roles:

- **caller**: an agent process started rote (Claude Code's Bash tool, Codex,
  Gemini CLI). Nobody is at the keyboard, so prompts are off and ``rote
  setup`` prints a plan instead (:mod:`rote.interact.mode`).
- **host**: rote runs inside a client's terminal, which may be a person
  typing. ``CLAUDECODE=1`` is set in IDE terminals too, and ``ZED_TERM`` in
  every Zed terminal, so these say which client to offer first for an MCP
  import and nothing about who is typing.

The variables are the union of what gh (``internal/agents/detect.go``) and
Vercel (``@vercel/detect-agent``) check, read from their source in
2026-10, plus the ones the first-run design found per client. Entries
marked unverified come from one third-party source; a wrong one costs a
prompt that should have been a plan, never a write, since every write is
confirmed either way.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

Role = Literal["caller", "host"]


@dataclass(frozen=True)
class Signal:
    """One environment variable that names a client."""

    var: str
    client: str
    role: Role
    #: The value it must have. None: any non-empty value.
    value: str | None = None
    verified: bool = True


#: Client ids are the ``ClientSource.id`` values in :mod:`rote.clients`.
SIGNALS: tuple[Signal, ...] = (
    # Claude Code: CLAUDECODE is set for the Bash tool, hooks, stdio MCP
    # servers and IDE terminals; CLAUDE_CODE_CHILD_SESSION only for direct
    # tool and hook subprocesses (2.1.172+, code.claude.com/docs/en/env-vars).
    Signal("CLAUDE_CODE_CHILD_SESSION", "claude-code", "caller", value="1"),
    Signal("CLAUDE_CODE_IS_COWORK", "claude-code", "caller"),
    Signal("CLAUDECODE", "claude-code", "host", value="1"),
    Signal("CLAUDE_CODE", "claude-code", "host", value="1"),
    Signal("CODEX_SANDBOX", "codex", "caller"),
    Signal("CODEX_CI", "codex", "caller"),
    Signal("CODEX_THREAD_ID", "codex", "caller"),
    Signal("CURSOR_AGENT", "cursor", "caller"),
    Signal("CURSOR_TRACE_ID", "cursor", "host"),
    Signal("GEMINI_CLI", "gemini", "caller"),
    Signal("COPILOT_CLI", "copilot", "caller"),
    Signal("OPENCODE", "opencode", "caller"),
    Signal("OPENCODE_CLIENT", "opencode", "caller"),
    Signal("AMP_CURRENT_THREAD_ID", "amp", "caller", verified=False),
    Signal("AGENT", "amp", "caller", value="amp"),
    # Set in a person's own shell by Goose's setup, so it says nothing about
    # who is typing.
    Signal("GOOSE_PROVIDER", "goose", "host"),
    Signal("AGENT_SESSION_ID", "goose", "caller", verified=False),
    Signal("ZED_TERM", "zed", "host"),
    # Set in every Replit shell, a person's included.
    Signal("REPL_ID", "replit", "host"),
)

#: ``AI_AGENT`` values (the emerging cross-vendor convention) that name a
#: client rote has a source for, matched as a prefix: Claude Code 2.1.302
#: sets ``AI_AGENT=claude-code_2-1-302_agent``. Any other value is still a
#: caller, named by the value itself.
AI_AGENT_CLIENTS: dict[str, str] = {
    "github_copilot_vscode_agent": "vscode",
    "opencode": "opencode",
    "claude-code": "claude-code",
    "claude_code": "claude-code",
    "codex": "codex",
    "cursor": "cursor",
    "gemini-cli": "gemini",
    "amp": "amp",
    "goose": "goose",
}


def _ai_agent_client(value: str) -> str:
    lowered = value.lower()
    for prefix, client in AI_AGENT_CLIENTS.items():
        if lowered == prefix or (lowered.startswith(prefix) and lowered[len(prefix)] in "_-/@ "):
            return client
    return value


@dataclass(frozen=True)
class Detected:
    """The client in the environment, the role it plays, and the variable that said so."""

    client: str
    role: Role
    var: str

    @property
    def label(self) -> str:
        return f"{self.client} ({self.var})"


def detect(env: Mapping[str, str]) -> Detected | None:
    """The strongest signal in ``env``: ``AI_AGENT``, then any caller, then any host."""
    ai_agent = env.get("AI_AGENT", "").strip()
    if ai_agent:
        return Detected(_ai_agent_client(ai_agent), "caller", "AI_AGENT")
    for role in ("caller", "host"):
        for signal in SIGNALS:
            if signal.role != role:
                continue
            value = env.get(signal.var, "")
            if not value:
                continue
            if signal.value is not None and value != signal.value:
                continue
            return Detected(signal.client, signal.role, signal.var)
    return None
