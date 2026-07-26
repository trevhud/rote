# Agent Runtime Decision Record

**Status:** accepted
**Date:** 2026-04-07
**Task:** #12

## Context

`rote` compiles AI skills into deterministic workflows. The compiler
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

**rote ships with three interchangeable compiler drivers, all
implementing the same Protocol:**

| Driver | Runtime | Auth | Install cost |
|---|---|---|---|
| `claude` | subprocess: `claude -p` | Claude Max/Pro OAuth or `CLAUDE_CODE_OAUTH_TOKEN` | user installs Claude Code separately |
| `codex` | subprocess: `codex exec` | ChatGPT Plus/Pro OAuth | user installs Codex CLI separately |
| `api` | in-process: `anthropic` SDK | `ANTHROPIC_API_KEY` | `pip install rote-cli[api]` |

Auto-detection order when `--agent` is not specified:
**`claude` → `codex` → `api`**. The CLI surface:

```sh
rote compile ./skill --runtime temporal --out ./compiled/          # auto-detect
rote compile ./skill --agent claude ...                              # explicit
rote compile ./skill --agent codex ...
rote compile ./skill --agent api ...
```

## The output contract (unifying the three drivers)

All drivers contract on the **filesystem**, not stdout parsing:

1. rote creates a scratch `work_dir/`.
2. The driver is told: "run the compiler agent against `skill_dir/`,
   using the rubric at `compiler_skill_dir/`, and the agent's final
   deliverable is `work_dir/pipeline.yaml`."
3. The driver returns after the agent exits (or the in-process loop
   completes). rote reads `work_dir/pipeline.yaml` and validates it
   with `rote.ir.load_pipeline`.
4. The agent may also write `work_dir/extracted/*.py`,
   `work_dir/signatures/*.py`, etc. — those land in the compiled
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
    --add-dir "<compiler_skill_dir>" \
    --add-dir "<work_dir>" \
    --allowedTools "Read,Write,Edit,Glob,Grep" \
    --output-format json \
    --max-turns 60
```

**Model selection — this matters a lot.** Claude Code's interactive
default is Opus 4.6, which is expensive overkill for the compiler's
structured-rubric task. The first two real compilations on BDR with
Opus burned ~$3.50 each in subscription accounting (2.6M cache-read
tokens per run, 31 turns, ~8 minutes) and exhausted the Max/Pro
"extra usage" budget in two runs. Switching to Sonnet 4.6 drops per-
run cost to ~$0.70, which makes iterative rubric tuning feasible.
rote defaults to ``claude-sonnet-4-6`` for that reason; users can
override with ``rote compile --model claude-opus-4-6`` for complex
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

**Skill injection:** the rote-compile SKILL.md is passed via
`--append-system-prompt` rather than relying on Claude's skill
auto-discovery from `.claude/skills/`. This is deterministic and
doesn't depend on the user installing rote's skill to a global
location. The source skill files are read by the agent via the
file-reading tools once we `--add-dir` them.

**Env flags:** set `CLAUDE_CODE_DISABLE_NONINTERACTIVE_ANIMATIONS=1`
to silence progress spinners in the subprocess output.

**Output capture:** `--output-format json` emits one JSON object with
`result`, `cost_usd`, `duration_ms`, `num_turns`, `session_id`. rote
captures this for the compilation report. The actual `pipeline.yaml`
comes from the filesystem, not stdout.

### `codex` (primary target for ChatGPT users)

**Invocation:**

```sh
codex exec \
    --cd "<work_dir>" \
    --sandbox workspace-write \
    --skip-git-repo-check \
    --ephemeral \
    --color never \
    --output-last-message "<tempfile outside work_dir>" \
    [--model "<id>"] \
    "<prompt with inlined SKILL.md>"
```

This is the **verified** invocation (codex-cli 0.142.4, confirmed by
both `codex exec --help` and a live smoke test through `CodexDriver`).
Three flags from the original design were wrong and were removed:

- **`--add-dir` dropped.** It appends to the sandbox's *writable* roots
  — it does not grant read access, and the source skill must stay
  read-only. Under `--sandbox workspace-write`, reads are already
  full-access across the filesystem, so the agent reads the skill and
  rubric in place with no grant.
- **`--ask-for-approval never` dropped.** It is not a valid flag on
  `codex exec` (it hard-errors). `exec` is headless: its approval
  policy already defaults to `Never`, so nothing is needed.
- **`--json` dropped** in favor of `--output-last-message`. The
  deliverable is a file on disk; the last-message file captures a clean
  final message for metadata without parsing an NDJSON event stream.
  It's written *outside* `work_dir` so the orchestrator's move of
  `work_dir` → user output doesn't sweep it up.

**Sandbox default gotcha:** `codex exec` defaults to a **read-only**
sandbox, which cannot write the pipeline.yaml — passing
`--sandbox workspace-write` is mandatory, not optional.

**Auth:** `codex exec` reuses the cached login from `~/.codex/auth.json`
(`codex login` once with a ChatGPT account; tokens auto-refresh). The
driver passes the environment through **untouched** — no scrubbing.
Unlike Claude (where `ANTHROPIC_API_KEY` overrides the OAuth session),
a stored Codex login is only overridden by `CODEX_API_KEY` /
`CODEX_ACCESS_TOKEN` — *not* by `OPENAI_API_KEY`. And because rote has
no OpenAI-API driver, a user whose only credential is `OPENAI_API_KEY`
must still be able to compile through Codex, so forcing subscription
auth would be wrong here.

**Git repo requirement:** by default Codex insists the workspace be a
git repo; rote's scratch `work_dir/` isn't, so `--skip-git-repo-check`
is required.

**Skill injection:** Codex has no `--append-system-prompt` flag, so the
rote-compile SKILL.md is inlined directly into the prompt argument
(its reference files are read from disk under the sandbox's global read
access). As with Claude, the `pipeline.yaml` comes from the filesystem.

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
class CompilerDriver(Protocol):
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
        compiler_skill_dir: Path,
        work_dir: Path,
    ) -> DriverResult:
        """Run the compiler agent and return a DriverResult.

        Raises DriverError on any failure (CLI missing, auth missing,
        agent crashed, pipeline.yaml not produced, etc.).
        """
        ...
```

`DriverResult` carries the path to the produced `pipeline.yaml`
plus metadata (token counts, cost estimate, duration, driver name)
for the compilation report.

## References

- [Claude Code headless docs](https://code.claude.com/docs/en/headless)
- [Claude Code authentication](https://code.claude.com/docs/en/authentication)
- [Claude Code print-mode auth gotcha](https://github.com/anthropics/claude-code/issues/3040)
- [Codex CLI non-interactive mode](https://developers.openai.com/codex/noninteractive)
- [Codex CLI reference](https://developers.openai.com/codex/cli/reference)
- [Codex authentication](https://developers.openai.com/codex/auth)
- [anthropic Python SDK on PyPI](https://pypi.org/project/anthropic/)
- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview)
