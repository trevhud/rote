---
name: implement
description: >-
  Fill in the stub modules of a pipeline emitted by rote, so it actually runs.
  Use after `rote emit` or `rote compile` when the emitted app still has
  extracted/* stubs that raise a "[rote stub]" error or a bare
  NotImplementedError, or when the user says "implement the
  stubs", "fill in the extracted modules", "make the compiled pipeline run",
  "wire up the real API calls", or `rote run` fails on a stub. Reads each
  node's IR contract (inputs, declared output keys, constants, description,
  source section), writes real code that honors it, leaves rote-owned files
  alone, and verifies with `rote run`. Spends no compilation tokens.
argument-hint: "[emitted-dir]"
---

# Implement the stubs of an emitted pipeline

`rote emit` and `rote compile` produce a runnable app whose deterministic
steps are scaffolds: every `pure_function` and `external_call` node that
has no real implementation yet gets a stub that fails on call. Your job is
to replace those stubs with working code that honors each node's IR
contract, then prove the pipeline runs end to end with `rote run`.

You write code only in user-owned files. rote owns everything else and
regenerates it on the next emit.

The CLI ships on PyPI as `rote-cli`. Invoke it as
`uvx --from rote-cli rote <args>`, which works on every release (the
`uvx rote-cli` shorthand needs a release newer than 0.14.0). Running a
Python app needs that runtime's extra in the same environment, because
`rote run` executes it with its own interpreter: `uvx --from 'rote-cli[dbos]' rote run ...` for DBOS,
`'rote-cli[temporal]'` for Temporal. If the user is inside a rote
checkout with a `.venv`, `.venv/bin/rote` works too.

## 1. Locate the emitted dir and its IR

The user names the directory, or it is the most recent `--out` in the
conversation. Accept any of:

- a `rote compile --out` dir: IR at `compiled/pipeline.yaml`, apps under
  `runtime/<target>/` (ask which target if there are several);
- a bare emitted runtime dir.

Find the runtime from its marker files, in this order: `wrangler.jsonc`
= cloudflare, `src/inngest/` = inngest, `dbos-config.yaml` + `main.py` =
dbos, `dbos-config.yaml` alone = dbos-ts, `workflow.py` + `activities.py`
= temporal, `main.py` alone = python.

Find the IR the way `rote run` does: `<dir>/pipeline.yaml`, then
`<dir>/../../compiled/pipeline.yaml` when the parent is named `runtime`,
then `<dir>/../pipeline.yaml`. If none exists, ask the user for it. Do not
implement from the emitted code alone: the IR is the contract, and the
stub docstrings are a lossy rendering of it.

Confirm the directory, runtime, and IR path with the user before editing.

## 2. Build the TODO list

Get the list from rote rather than by guessing. Re-emitting into the same
directory is safe (see section 6) and reports what is still a stub:

```sh
uvx --from rote-cli rote emit <pipeline.yaml> --runtime <runtime> --out <dir> --json
```

Read three keys from the JSON:

- `unimplemented_stubs`: absolute paths of modules that still have a
  stub function. This is the TODO list. A module drops off it once none
  of its functions is a stub. In Python that means no function raises
  with the `[rote stub]` sentinel and none has a body that is only
  `raise NotImplementedError`. In TypeScript it means `[rote stub]`
  appears nowhere outside a comment. The full rule, with examples, is in
  [references/runtimes.md](references/runtimes.md#what-counts-as-a-stub).
- `preserved_new_files`: `.new` siblings parked next to files the user
  already edited. Handle these before anything else (section 6).
- `contract_findings`: IR promises the implementations beside the
  `pipeline.yaml` break (missing function, wrong signature, syntax
  error). Treat any `error` as a defect to fix.

The list covers every runtime, temporal included. Files under
`extracted/` that are not on it are rote's, even when they sit right
next to the stubs (see section 5).

## 3. Read each node's contract

For every stub, find every node whose `impl` points at it (several nodes
can share one Python function). Read from `pipeline.yaml`:

| Field | What it tells you |
|---|---|
| `kind` | `pure_function` = no I/O, deterministic. `external_call` = network I/O |
| `description` | what the step does, in the compiler's words |
| `input` / `inputs` | the parameter names, their types, and where each value comes from |
| `output` | a single type name (return that value bare) or a key-to-type map (return a dict with exactly those keys) |
| `constants` | values lifted from the source skill: thresholds, IDs, queries |
| `mandatory` | the workflow always calls this node, so never make it conditional |
| `timeout` / `retry` | the runtime retries, so your code should raise on failure, not swallow it |
| `source.section` | the heading in the original SKILL.md this node came from |

Then open the source skill: `source_skill` in the IR is a path relative
to the `pipeline.yaml`. Read the section each node names. It holds the
detail the IR summarized: column names, the exact rule behind a
threshold, report formatting. When the description and the source
disagree, the source skill is what the user ran in production. Follow it
and tell the user about the gap.

Work out the data shapes by tracing the edges. A parser's input is its
upstream fetcher's output, and the final node's input is every parser's
output. Implement the producer and consumer of each edge against the
same shape. The exact call shape (keyword arguments for Python, one
input object for TypeScript) is in [references/runtimes.md](references/runtimes.md).
Read the generated call site in the workflow file to confirm it.

Two rules that are easy to get wrong:

- **Constants: Python receives them, TypeScript does not.** On dbos,
  python and temporal, each node's IR `constants` arrive as keyword
  arguments next to its inputs. A function several nodes share
  receives the calling node's values, so take each constant as a
  keyword parameter and never hard-code one node's value. Every calling
  node is listed in the stub's docstring with its constants. A constant
  that only some callers send needs a default. The TypeScript runtimes
  pass no constants. Each TS stub serves exactly one node, and its doc
  comment lists that node's constants, so write them into that file as
  named constants. Confirm either way against the generated call site.
- **Optional pipeline inputs may be absent.** An `input_schema` field not
  in `required` can be missing or null. Apply the default the schema's
  description names (for example "omit to default to today") inside the
  node that consumes it.

## 4. Write the implementations

Write real code. No placeholders, no `TODO`, no sample data presented as
real output, no silent fallbacks: a failure should raise with a message
that names the node, because the runtime's retry and error reporting
depend on it.

- Keep the function name and module path exactly as the IR's `impl` (or
  the TS export the workflow imports) names them. The workflow file
  imports them by that name.
- `pure_function`: deterministic, no network, no clock reads unless the
  contract asks for one (a "defaults to today" input is such a case).
  Return the declared output keys and nothing the contract does not
  promise.
- `external_call`: call the vendor API directly with its official SDK or
  plain HTTP. Read credentials from environment variables, name them in
  the module docstring, and fail with a clear error when one is missing.
  Look up the vendor's current API docs before writing the call. Do not
  rely on memory for endpoints or SDK shapes. Honor the IR `timeout` on
  the request itself where the runtime has no per-step timeout (DBOS,
  python).
- An `external_call` with an `mcp:` binding is not a stub. rote emits a
  working MCP call for it. Leave it alone.
- Add any new dependency where the emitted app declares them
  (`requirements.txt`, `package.json`) and tell the user.

If a node needs credentials or a service nobody has here, ask the user
how they want to proceed. When they want a local run anyway, implement
the real client and select a local fixture source with an explicit
environment variable (for example a directory of captured responses),
so production keeps the real path and the fixture path is visible in
the code. Say in your summary which nodes ran against fixtures.

## 5. Never edit rote-owned files

These are regenerated on every emit. Edits there are either overwritten
or get parked as conflicts, and some are byte-compared in rote's own
tests:

- the workflow and wiring: `main.py` (dbos, python), `workflow.py`,
  `activities.py` and `__init__.py` (temporal), `src/workflow.ts`,
  `src/index.ts`, `src/main.ts`, `src/inngest/*`;
- generated agent-loop modules on the TypeScript runtimes:
  `src/extracted/<agent_loop node id>.ts`. They sit beside the stubs
  but are working code that binds the loop's tools and calls the
  inference helper;
- the verbatim runtime helpers: `extracted/_rote_mcp.py`,
  `signatures/_rote_inference.py`, `src/extracted/_roteMcp.ts`,
  `src/signatures/_roteInference.ts`;
- judge modules (`signatures/*`, `src/signatures/*`) and
  `extracted/__init__.py`;
- config and metadata: `dbos-config.yaml`, `wrangler.jsonc`,
  `tsconfig.json`, `manifest.json`, `.dev.vars.example` (copy it to
  `.dev.vars` for local secrets), `README.md`, and
  `.rote-manifest.json` (never edit or delete it, because it is how
  re-emit knows which files are yours).

If the bug is in one of these files, the fix belongs in the IR or in
rote itself. Stop and tell the user. Do not patch emitted code.

Your files are exactly the stub modules on step 2's list, plus any new
modules you import from them.

## 6. Respect re-emit

rote's emit writer records the hash of every file it writes in
`.rote-manifest.json`. On the next emit it overwrites a file only if it
is still exactly what rote wrote. A file you edited is preserved, and the
fresh rote output lands next to it as `<name>.new`.

- A `.new` sibling of a file you implemented is usually the regenerated
  stub. Read the diff: if only the docstring changed, delete the `.new`.
  If the contract changed (new parameter, new output key, new
  constant), update your implementation to match, then delete the
  `.new`. Expect it back after every emit: rote keeps parking the fresh
  stub beside a file you own, so a `.new` that is only the stub again is
  safe to delete each time.
- A `.new` next to a rote-owned file means someone edited that file.
  Show the user the diff. The usual answer is to move their change
  somewhere it belongs and accept the fresh file.
- For dbos, python and temporal, emission copies
  `extracted/<module>.py` from beside the `pipeline.yaml` into the
  runtime dir verbatim. When that
  directory exists, it is the source of truth: write the implementation
  there and re-emit, so every Python runtime emitted from this IR gets
  the same code.

## 7. Verify with `rote run`

Always pass `--input`. Without it, `rote run` derives an input from the
source skill with a model call, which spends tokens and prompts for
confirmation. Build the input from the IR's `input.input_schema`.

Every HITL gate needs a resume payload up front, one `--signal` per gate,
shaped like the gate's `output` in the IR:

```sh
uvx --from 'rote-cli[dbos]' rote run <dir> --json \
  --input '{"report_date": "2026-07-18"}' \
  --signal manual_data_provided='{"sorter_metrics": [], ...}'
```

`rote run` refuses to start while any module on step 2's list is still a
stub, and names them. Fix those first. Pass `--allow-stubs` only when
the run cannot reach the remaining stubs (they sit on a branch this input
does not take), and say so in your report. A stub the run does reach
still raises.

A zero exit with `"error": null` and an `output` keyed by the exit nodes
is a pass. On failure, read the error, fix the implementation, and run
again. If the failure is inside a rote-owned file, follow section 5.

Also run the project's own checks where they exist: the emitted test
suite if there is one, `tsc --noEmit` after `npm install` for TypeScript
runtimes.

## 8. Report

Tell the user, in this order:

1. what `rote run` returned (pass or fail, wall time, and the output's
   shape);
2. each implemented node: what it calls, which credentials it reads, and
   whether it ran against real services or fixtures;
3. any contract gaps you found (description vs source skill, an output
   key no producer can supply) and anything left unimplemented, with why;
4. new dependencies and environment variables they need to set before
   deploying.
