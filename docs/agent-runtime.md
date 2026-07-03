# Agent Runtime Decision Record

**Status:** accepted
**Date:** 2026-04-07
**Task:** #12

## Context

`rote` graduates AI skills into deterministic workflows. The graduator
phase — reading a source skill, producing a `pipeline.yaml`, stubbing
extracted modules and signatures — requires an LLM agent loop. That
agent has to run *somewhere*.

Most rote users will be engineers who already pay for a coding agent:
a Claude Max/Pro subscription (used via Claude Code), a ChatGPT
Plus/Pro subscription (used via Codex CLI), or an Anthropic API key.
Asking them to also configure per-token API billing just to use rote
is unnecessary friction, and it rules out the "free for subscription
users" sweet spot that an OSS dev tool needs.

## Decision

**rote ships with three interchangeable graduator drivers, all
implementing the same Protocol:**

| Driver | Runtime | Auth | Install cost |
|---|---|---|---|
| `claude` | subprocess: `claude -p` | Claude Max/Pro OAuth or `CLAUDE_CODE_OAUTH_TOKEN` | user installs Claude Code separately |
| `codex` | subprocess: `codex exec` | ChatGPT Plus/Pro OAuth | user installs Codex CLI separately |
| `api` | in-process: `anthropic` SDK | `ANTHROPIC_API_KEY` | `pip install rote-cli[api]` |

Auto-detection order when `--agent` is not specified:
**`claude` → `codex` → `api`**. The CLI surface:

```sh
rote graduate ./skill --runtime temporal --out ./graduated/          # auto-detect
rote graduate ./skill --agent claude ...                              # explicit
rote graduate ./skill --agent codex ...
rote graduate ./skill --agent api ...
```

## The output contract (unifying the three drivers)

All drivers contract on the **filesystem**, not stdout parsing:

1. rote creates a scratch `work_dir/`.
2. The driver is told: "run the graduator agent against `skill_dir/`,
   using the rubric at `graduator_skill_dir/`, and the agent's final
   deliverable is `work_dir/pipeline.yaml`."
3. The driver returns after the agent exits (or the in-process loop
   completes). rote reads `work_dir/pipeline.yaml` and validates it
   with `rote.ir.load_pipeline`.
4. The agent may also write `work_dir/extracted/*.py`,
   `work_dir/signatures/*.py`, etc. — those land in the graduated
   output directory alongside the IR.

This means every driver has the same interface and every rote user
gets the same output shape regardless of which backend ran the agent.

## Driver details

### `claude` (primary target for subscription users)

**Invocation:**

```sh
claude -p "<prompt>" \
    --model claude-sonnet-4-6 \
    --append-system-prompt "<inlined SKILL.md>" \
    --add-dir "<skill_dir>" \
    --add-dir "<graduator_skill_dir>" \
    --add-dir "<work_dir>" \
    --allowedTools "Read,Write,Edit,Glob,Grep" \
    --output-format json \
    --max-turns 60
```

**Model selection — this matters a lot.** Claude Code's interactive
default is Opus 4.6, which is expensive overkill for the graduator's
structured-rubric task. The first two real graduations on BDR with
Opus burned ~$3.50 each in subscription accounting (2.6M cache-read
tokens per run, 31 turns, ~8 minutes) and exhausted the Max/Pro
"extra usage" budget in two runs. Switching to Sonnet 4.6 drops per-
run cost to ~$0.70, which makes iterative rubric tuning feasible.
rote defaults to ``claude-sonnet-4-6`` for that reason; users can
override with ``rote graduate --model claude-opus-4-6`` for complex
skills where Opus's extra reasoning earns its cost.

**Turn budget.** BDR-scale skills need ~25 tool calls minimum and
realistically 40–50 with exploration. The default ``--max-turns 60``
leaves headroom; 30 is not enough and produced an ``error_max_turns``
on our first attempt.

**Auth gotcha (critical):** in `claude -p` mode, the
`ANTHROPIC_API_KEY` env var *always wins* over any active OAuth
session. To use subscription auth, the subprocess environment must
not contain `ANTHROPIC_API_KEY` or `ANTHROPIC_AUTH_TOKEN`. The
`ClaudeDriver` always scrubs these env vars before spawning so that
Claude falls back to the user's interactive login. If the user
genuinely wants API-key auth, they should use the `api` driver
(which is the one explicit place rote honors `ANTHROPIC_API_KEY`).

The alternative is `CLAUDE_CODE_OAUTH_TOKEN` — a long-lived OAuth
token Anthropic issues for automation. If the user sets this in
their env, `ClaudeDriver` passes it through. This is the cleanest
path for CI environments where no interactive `claude login` has
been run.

**Skill injection:** the rote-graduate SKILL.md is passed via
`--append-system-prompt` rather than relying on Claude's skill
auto-discovery from `.claude/skills/`. This is deterministic and
doesn't depend on the user installing rote's skill to a global
location. The source skill files are read by the agent via the
file-reading tools once we `--add-dir` them.

**Env flags:** set `CLAUDE_CODE_DISABLE_NONINTERACTIVE_ANIMATIONS=1`
to silence progress spinners in the subprocess output.

**Output capture:** `--output-format json` emits one JSON object with
`result`, `cost_usd`, `duration_ms`, `num_turns`, `session_id`. rote
captures this for the graduation report. The actual `pipeline.yaml`
comes from the filesystem, not stdout.

### `codex` (primary target for ChatGPT users)

**Invocation:**

```sh
codex exec \
    --cd "<work_dir>" \
    --add-dir "<skill_dir>" \
    --add-dir "<graduator_skill_dir>" \
    --sandbox workspace-write \
    --ask-for-approval never \
    --skip-git-repo-check \
    --json \
    "<prompt with inlined SKILL.md>"
```

**Auth:** `codex exec` reuses the cached login from
`~/.codex/auth.json`. User runs `codex login` once with their
ChatGPT account; tokens auto-refresh. No env scrubbing needed.

**Gotcha — git repo requirement:** by default, Codex insists that
the workspace directory be a git repo. For rote's scratch `work_dir/`
this is not the case, so `--skip-git-repo-check` is required.

**Gotcha — auth conflict:** an active ChatGPT login *blocks*
`OPENAI_API_KEY` from being used. rote doesn't try to support both
auth modes through the codex driver — if the user wants API-key
auth, they should use the `api` driver (which is Anthropic, not
OpenAI, and a different choice entirely).

**Skill injection:** Codex has no `--system-prompt` flag, so the
rote-graduate SKILL.md is inlined directly into the prompt argument.
Codex does read `~/.codex/skills/` but rote does not rely on that.

**Output capture:** `--json` emits NDJSON events on stdout. Progress
goes to stderr. As with Claude, the `pipeline.yaml` comes from the
filesystem.

### `api` (for users who prefer API key auth)

**Implementation:** in-process tool-use loop using the bare
`anthropic` Python SDK (latest: 0.89.0 as of 2026-04-03). The
driver implements two minimal tools — `read_file` scoped to the
source skill directory, and `write_file` scoped to the work dir —
and runs the standard Messages API tool-use loop until the model
stops calling tools.

**Dependency:** the `anthropic` SDK is an optional extra. Users who
only want the subprocess drivers never pay the import cost. Install:

```sh
pip install "rote-cli[api]"
```

**Auth:** `ANTHROPIC_API_KEY` environment variable. The driver
errors out early with a helpful message if the env var is missing.

## Explicit non-goals

### Not using `claude-agent-sdk`

**Anthropic's terms of service explicitly forbid third-party agents
built on the Claude Agent SDK from using claude.ai login credentials
without prior approval.** The Agent SDK is an API-key-only path for
third-party tooling per policy, not just technical limitation.

This means the Agent SDK offers rote nothing that the bare `anthropic`
SDK doesn't — and it brings a heavier dependency footprint plus a
subprocess wrapper around `claude` that `ClaudeDriver` already does
more directly. We explicitly do not depend on `claude-agent-sdk`.

### Not supporting every coding agent CLI

rote does not ship drivers for Aider, Gemini CLI, Cursor Agent,
Continue, etc. The three chosen drivers cover the two dominant
subscription paths (Claude, ChatGPT) plus the direct API path.
Community contributions for other drivers are welcome once the
Protocol is stable; they should not live in the core package.

### Not parsing stdout for structured output

Every driver contracts on `work_dir/pipeline.yaml` being the
deliverable. We do not parse streaming JSON events for structured
output. The filesystem contract is:
- simpler (no format-specific parsers per driver),
- debuggable (the user can inspect the work dir),
- uniform across in-process and subprocess drivers.

## Auto-detect behavior

When the user does not pass `--agent`:

1. Check `claude` CLI presence (`shutil.which("claude")`). If found,
   use it. Rationale: Claude Max/Pro subscription is the most common
   path for rote's expected audience (engineers already using Claude
   Code).
2. Else check `codex` CLI presence. If found, use it.
3. Else check that `anthropic` is importable AND `ANTHROPIC_API_KEY`
   is set. If both, use the api driver.
4. Else fail with a helpful error listing all three install/setup
   options.

User passes `--agent <name>` to override. Authority is explicit user
choice > auto-detect.

## The Protocol

```python
class GraduatorDriver(Protocol):
    name: str

    def is_available(self) -> tuple[bool, str]:
        """(available, reason_if_not).
        When available is False, reason_if_not is a user-facing
        message explaining how to fix it.
        """
        ...

    async def run(
        self,
        skill_dir: Path,
        graduator_skill_dir: Path,
        work_dir: Path,
    ) -> DriverResult:
        """Run the graduator agent and return a DriverResult.

        Raises DriverError on any failure (CLI missing, auth missing,
        agent crashed, pipeline.yaml not produced, etc.).
        """
        ...
```

`DriverResult` carries the path to the produced `pipeline.yaml`
plus metadata (token counts, cost estimate, duration, driver name)
for the graduation report.

## References

- [Claude Code headless docs](https://code.claude.com/docs/en/headless)
- [Claude Code authentication](https://code.claude.com/docs/en/authentication)
- [Claude Code print-mode auth gotcha](https://github.com/anthropics/claude-code/issues/3040)
- [Codex CLI non-interactive mode](https://developers.openai.com/codex/noninteractive)
- [Codex CLI reference](https://developers.openai.com/codex/cli/reference)
- [Codex authentication](https://developers.openai.com/codex/auth)
- [anthropic Python SDK on PyPI](https://pypi.org/project/anthropic/)
- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview)
