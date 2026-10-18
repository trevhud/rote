# How rote compares

Several tools now turn an agent skill into something more structured
than a prompt. They solve different problems. This page compares rote
with the ones you are most likely to be choosing between, says where
each one is the better choice, and cites the source for every claim
about another project.

Facts about other projects were checked on 2026-10-18 against the
versions named below. They move fast, so check the linked sources
before relying on a detail.

| | rote | calcifier | skill-to-workflow | Claude Code dynamic workflows | Hand-written workflow |
| --- | --- | --- | --- | --- | --- |
| Input | `SKILL.md` + `references/` | `SKILL.md` + resources | `SKILL.md` + bundled files | A task you describe. Claude writes the script | Your design |
| Output | Runtime-agnostic IR (`pipeline.yaml`) plus emitted code | One Python recipe module (`.calcifier/recipe.py`) | One Claude Code workflow script (`*.workflow.js`) | One workflow script (`.claude/workflows/*.js`) | Whatever you write |
| Where it runs | DBOS (Python or TS), Temporal, Cloudflare Workflows, Inngest, or a plain Python script | Your host, via `calcifier exec` | Inside a Claude Code session | Inside a Claude Code session | Anywhere |
| What a judgment step is | One schema-locked model call with a typed signature. `agent_loop` only for open-ended tool use | A full Claude or Codex CLI agent session with declared file access, parsed into a typed value | A subagent | A subagent | Your choice |
| Deterministic steps | Code, no model involved | Python, no model involved | Still agents (the script has no shell or file access) | Still agents (same limit) | Code |
| Goal | Fewer model calls, lower cost, repeatable output | Fewer model calls, reviewable Python | More parallelism and adversarial verification, which adds agents | Orchestrate many agents | Whatever you need |
| Survives a crash or restart | Yes, through the runtime's durable execution | No durable resume. A `parallel()` callback can checkpoint results | Resumable in the same Claude Code session | Resumable in the same session | Depends on your runtime |
| Human approval mid-run | `hitl_gate` nodes that suspend durably and resume on a signal | Not built in | No mid-run input | No mid-run input. Split stages into separate workflows | You build it |
| MCP tools | Real MCP client calls in emitted code. A missing or expired credential parks the run until `rote mcp login` releases it | Agent sessions run with `--strict-mcp-config` and no MCP servers. External tools go through subprocess tool providers | Through subagents | Through subagents | You wire it |
| Per-item fan-out | `fan_out` nodes on all six runtimes | `parallel()` | `pipeline()` / `parallel()` | `pipeline()` / `parallel()` | You build it |
| Cost and determinism report | `rote eval` scorecard (estimate) and `rote eval --run` (measured, both sides) | `calcifier profile` summarizes a run's timings, attempts, and tokens | Not provided | Per-agent token usage in `/workflows` | You build it |
| Re-compile after the skill changes | `rote compile --update` re-derives only changed sections and keeps node ids and filled-in code | `calcifier update` recompiles and prints a diff without writing | Re-run the converter | Ask Claude to edit the script | You edit it |
| Who pays for model calls | Chosen at run time per node: Claude subscription, your API key, or rote cloud | The configured Claude or Codex CLI, or a custom agent command | Your Claude Code session | Your Claude Code session | Your choice |
| Compile-time checks | Pydantic IR validation, deterministic emission, emitted code tested per runtime in CI | Ruff, ty, semantic and taint checks, and a whole-recipe dry run with stub agents | A V8 parse gate plus three review agents | Schema contradictions caught before an agent starts | Your CI |
| License | Apache-2.0 | MIT | MIT | Part of Claude Code | n/a |

## Measured: compiling the same skill

On 2026-10-18 we compiled
[`examples/ops-report/skill`](../examples/ops-report/skill/SKILL.md)
with rote. The skill pulls three Google Drive files and one Gmail label
over MCP, asks a duty manager for three numbers, then applies fixed
thresholds and renders a report. Both tools were meant to run with
`claude-sonnet-4-6` through the `claude` CLI.

The calcifier side was not run. The compile command was stopped by a
local permission check before it started, so this table has no
calcifier numbers. It will be filled in from a real run, not estimated.

| | rote 0.14.0.dev0 | calcifier 0.1.1 |
| --- | --- | --- |
| Command | `rote compile examples/ops-report/skill --local --runtime dbos --agent claude --yes` | `calcifier calcify SKILL.md --agent-config agents.json` (both roles on the `claude` builtin, Sonnet 4.6) |
| Compile wall time | 747 s | not run |
| Compile cost | $2.43 (Claude CLI estimate, 37 turns: 51k output, 133k cache-write, 2.89M cache-read tokens) | not run |
| Output | 359-line `pipeline.yaml` with 11 nodes (4 MCP `external_call`, 6 `pure_function`, 1 `hitl_gate`, 0 `llm_judge`), a 538-line DBOS `main.py`, 5 extracted modules | not run |
| Runs as compiled | No. The 6 functions the workflow calls (4 parsers, 1 validator, 1 renderer) are `NotImplementedError` stubs with typed docstrings. The first run stopped earlier, on the missing MCP endpoints: `no endpoint for MCP server 'gmail' ... set ROTE_MCP_GMAIL_URL` | not run |
| Durability | DBOS workflow with SQLite system database. Steps are checkpointed and the run survives a process restart | not run |
| Human input | `DBOS.recv` on topic `duty_manager_data_submitted`, 4 h timeout from the IR, survives restarts | not run |
| Model calls per run afterwards | 0 (no `llm_judge` or `agent_loop` nodes). `rote eval` estimates the skill as an agent at $0.20 to $0.29 per run on the current mid-tier Sonnet. That number is an estimate, not a measurement | not run |

Two results here count against rote:

- **Compile cost.** $2.43 for a 50-line skill is about three times
  what the README quotes for the much larger BDR skill. Ten of the 37
  turns went to small edits of `compile-report.md`, each of which
  re-reads the full context.
- **Nothing runs until you write code.** The compiler classified every
  step correctly but did not implement the parsing, threshold, and
  rendering functions, even though the skill states each rule exactly.
  Calcifier's design accepts a recipe only after a whole-recipe dry run
  with stub agents, so a calcifier output that passes compilation does
  execute.

## rote

rote reads a skill and classifies every step into one of five node
kinds: `pure_function`, `external_call`, `llm_judge`, `agent_loop`, or
`hitl_gate`. The result is a `pipeline.yaml` that is validated by
[`src/rote/ir.py`](../src/rote/ir.py) and is not tied to any runtime.
Adapters turn that IR into native code for one of six targets
([`src/rote/adapters/`](../src/rote/adapters/)). Emission is template
substitution, so the same IR always produces the same files.

The trade-off is that rote asks more of you up front. You pick a
durable runtime and operate it (or use rote cloud), and an
`external_call` node that the compiler could not implement is a stub
you fill in. The TypeScript runtimes cannot reuse Python
implementations, so they always start from stubs.

## calcifier

[trailofbits/calcifier](https://github.com/trailofbits/calcifier)
(v0.1.1, first released 2026-10-14) compiles a skill into one Python
recipe: typed `@step` functions composed by a `@recipe` entry point
with ordinary control flow
([docs/architecture.md](https://github.com/trailofbits/calcifier/blob/main/docs/architecture.md)).
Judgment steps call `agent(role, prompt, output=...)`, which runs a
configured Claude or Codex CLI session with a declared working
directory and read-only paths by default, then parses the answer into
a typed value
([docs/providers.md](https://github.com/trailofbits/calcifier/blob/main/docs/providers.md)).

Where calcifier is the better choice:

- **Skills that read a local codebase or files.** A calcifier agent
  step is a full CLI session that can read the target directory. rote's
  `llm_judge` is a single call on the data you pass it. For code review
  and repo triage, calcifier's shape fits better.
- **Safety checks on generated code.** Every candidate goes through
  Ruff, ty, semantic checks, taint checks that stop agent output from
  choosing file paths or command arguments, and a whole-recipe dry run
  with stub agents in a separate interpreter. Recipes also need
  trust-on-first-use consent before `exec` runs them. rote has nothing
  equivalent to the taint check or the stub dry run.
- **A safe fallback.** If compilation fails, calcifier keeps a runnable
  baseline that hands the intact skill to one agent session, and labels
  it as such.
- **No infrastructure.** One Python file, run on your host. Nothing to
  deploy.
- **Composition.** Recipes can call child recipes, and `build-graph`
  compiles a dependency graph of skills in order.

Where rote does more: calcifier has no durable resume, no
human-approval step, no scheduling or deploy target, and no MCP client
in its agent sessions (`--strict-mcp-config` with no servers,
[`agent_providers.py` line 59](https://github.com/trailofbits/calcifier/blob/99dfff5/src/calcifier/agent_providers.py#L59)).
It needs Python 3.13 or newer. rote needs 3.11.

## skill-to-workflow

[democra-ai/skill-to-workflow](https://github.com/democra-ai/skill-to-workflow)
(v0.1.0, May 2026) is itself a Claude Code workflow that reads a skill
and writes another Claude Code workflow script. Its goal is the
opposite of rote's on cost: it finds independent steps to run in
parallel and adds adversarial verification, with several skeptic
agents per finding
([README](https://github.com/democra-ai/skill-to-workflow#what-it-does)).
The output has more agents than the skill had, not fewer.

Choose it when the skill's weakness is quality or wall-clock time, not
cost, and you are happy for it to run inside Claude Code. It installs
with one `npx` command and has no runtime dependencies.

## Claude Code dynamic workflows

[Dynamic workflows](https://code.claude.com/docs/en/workflows) are
JavaScript scripts that orchestrate subagents with `agent()`,
`parallel()`, and `pipeline()`. Claude writes the script for a task,
and you can save it as a command. Runs are resumable within the same
session and show per-agent token usage.

The documented limits matter for this comparison. The script has "no
direct filesystem or shell access", so a step that is just a command
still needs an agent to run it
([anthropics/claude-code#98566](https://github.com/anthropics/claude-code/issues/98566),
open as of 2026-10-18). There is "no mid-run user input", and runs live
inside a Claude Code session.

Choose dynamic workflows for work that is mostly judgment spread over
many items, such as audits, migrations, and research, run by someone
who is already in Claude Code. rote is for the opposite case: a
mostly mechanical procedure that should run unattended, outside a
session, with as few model calls as possible.

## Claude Managed Agents

[Managed Agents](https://platform.claude.com/docs/en/managed-agents/overview)
(beta) host the agent loop for you, with skills, MCP servers, sandboxes,
persistent sessions, and cron schedules. That solves where an agent
runs. It does not reduce how much of the work is an agent. The two can
work together: an `agent_loop` node in a rote pipeline is still an
agent, and the rest of the pipeline is code.

## Writing the workflow by hand

A good engineer writing the workflow directly will often get a better
result than any compiler, including rote. You understand the edge
cases, you choose the runtime, and nothing is generated.

rote is worth it when the cost of doing that by hand is the problem: a
team with many skills, skills that keep changing (`--update` keeps
node ids stable so in-flight workflows survive), or a need to compare
the compiled pipeline against the original skill on the same inputs
(`rote eval --run`). If you have one skill and the time to write it,
write it.

## Research

Two papers study the same idea. "Compiled AI" (Trooskens et al.,
[arXiv:2604.05150](https://arxiv.org/abs/2604.05150)) measures the
token, latency, and reproducibility gains from compiling LLM workflows
into code. TraceCompiler (El Yadouni and Li,
[arXiv:2608.02680](https://arxiv.org/abs/2608.02680)) compiles recorded
agent traces, rather than skill text, into mostly deterministic
workflows, and declines to compile when an irreversible side effect is
under-determined. No code release is linked from the abstract page.
