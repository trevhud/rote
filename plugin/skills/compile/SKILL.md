---
name: compile
description: >-
  Compile an Anthropic-style skill — a directory with a SKILL.md and optional
  references/ — into a deterministic, runnable workflow via the rote CLI. Use
  when the user says "compile this skill", "graduate this skill" (the retired
  name for the same operation), "make this skill deterministic", "make this
  skill faster/cheaper", "turn this skill into a workflow", "turn this skill
  into code", "harden this skill for production", or complains that a skill
  is slow, expensive, or unreliable as a background job. Output: a
  pipeline.yaml IR, extracted code modules, typed LLM-judge signatures, and
  runtime code for Temporal, Cloudflare Workflows, or DBOS.
---

# Compile a skill

You orchestrate the `rote` CLI. It runs an LLM compiler agent over a
source skill and emits a deterministic pipeline. Your job: resolve the
inputs, run the CLI, then interpret the output for the user. You never
classify nodes or write pipeline.yaml yourself — the CLI's agent does.

## 1. Identify the source skill

The source is a **directory containing a `SKILL.md`** (optionally a
`references/` folder). The user names it, or you infer it from context
(a skill just discussed, a path in the conversation, `.claude/skills/*`
or `skills/*` in the project).

**Confirm the resolved absolute path with the user before running.**
Compilation costs real time and tokens; never guess-and-go. If the
directory has no `SKILL.md`, stop and ask.

## 2. Pick a runtime target

Ask the user which runtime, with these tradeoffs (one line each):

| Runtime | Choose when | Emits |
|---|---|---|
| `dbos` | No infra to run — durability lives in SQLite/Postgres, runs anywhere Python runs | Python |
| `cloudflare` | You want serverless, fully managed execution on Cloudflare Workers | TypeScript |
| `temporal` | You already operate (or want) a Temporal cluster | Python |

If the user has no opinion and no existing infra, use `dbos` — it is
the CLI's default and the only target with zero standing
infrastructure (you can omit `--runtime` entirely in that case).

## 3. Resolve the CLI (uv)

The CLI ships on PyPI as the `rote-cli` package and is run via `uvx` —
no virtualenv, no pip, nothing to install beyond uv itself. The
package's executable is named `rote`, so every invocation is
`uvx --from rote-cli rote <args>`. Do **not** run `uvx rote-cli ...` —
uvx looks for an executable named after the package and the published
wheel doesn't ship one.

1. Check uv: `uv --version`. If missing, tell the user to install it
   with one command, then re-check:

   ```sh
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Confirm the CLI resolves:

   ```sh
   uvx --from rote-cli rote --version
   ```

3. Only if the user needs unreleased features (or PyPI is
   unreachable), substitute the GitHub source — same CLI, different
   origin:

   ```sh
   uvx --from git+https://github.com/trevhud/rote rote --version
   ```

Do **not** clone the repo or build a venv; `uvx` handles isolation.

## 4. Run the compilation

```sh
uvx --from rote-cli rote compile <skill-dir> --runtime <runtime> --out <out-dir>
```

Pick an out-dir the user will find, e.g. `./compiled/<skill-name>`
next to the source skill. Ensure it does not clobber existing work.

Set expectations **before** launching — this is not a quick command:

- It spawns `claude -p` as a subprocess. The driver deliberately
  scrubs `ANTHROPIC_API_KEY` / `ANTHROPIC_AUTH_TOKEN` from the child
  environment so the run bills against the user's Claude
  subscription, not per-token API charges. Do not "fix" auth by
  exporting an API key; if the user explicitly wants API billing,
  pass `--agent api` instead.
- A realistic skill takes **~13 minutes wall clock and 30-40 agent
  turns** (Sonnet, ~$0.70 on subscription). Small skills are faster.
- Therefore **run it in the background** and tell the user you did.
  Poll the process and check in rather than blocking the session.

If the run exits nonzero, check whether `<out-dir>/compiled/pipeline.yaml`
exists anyway — the CLI recovers completed work from transient
subprocess failures and says so in its output. Surface stderr to the
user either way.

## 5. Report the result

Read `<out-dir>/compiled/pipeline.yaml` and
`<out-dir>/compiled/compile-report.md`, then summarize:

1. **Node-kind table** — count nodes per kind and what each kind means
   here:

   | Kind | Count | Meaning |
   |---|---|---|
   | `pure_function` | n | deterministic code, LLM removed |
   | `external_call` | n | direct API call with retry/timeout |
   | `llm_judge` | n | typed LLM signature (kept, but bounded) |
   | `agent_loop` | n | still agentic (genuinely exploratory) |
   | `hitl_gate` | n | durable human approval point |

2. **Codified fraction** — nodes that no longer need an LLM, mandatory
   nodes, and what each HITL gate blocks on.
3. **Where things landed** — `<out-dir>/compiled/` (IR, `extracted/`,
   `signatures/`, report) and `<out-dir>/runtime/<runtime>/` (the
   deployable code).
4. **Next steps** — the `extracted/*` modules are scaffolds that raise
   `NotImplementedError`; the user fills in real API client code, then
   deploys the runtime output. Once deployed, `rote register` +
   `rote serve` expose the pipeline as an MCP tool so Claude can
   trigger runs — the `serve` skill in this plugin walks through that.
