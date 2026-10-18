# Where stubs live, per runtime

Each runtime puts user code in a different place and calls it with a
different shape. Confirm against the generated workflow file every time.
That file is the contract the emitter actually wrote, and this page can
drift from it.

## Python runtimes (dbos, python, temporal)

- **Stubs:** `extracted/<module>.py`, one module per `impl` path. Several
  nodes can share one function (an IR `impl` of
  `extracted/google_drive.py:fetch_file_content` on three nodes is one
  function called three times).
- **Call shape:** the generated step calls `func(**payload)`. The payload
  keys are the node's `inputs:` parameter names plus its IR `constants`.
  Read the payload literal in `main.py` (dbos, python) or `workflow.py`
  (temporal) for the exact keys per node. Give the function explicit
  keyword parameters rather than `**payload`, so a contract change fails
  loudly. When nodes sharing a function send different constant sets
  (one sends `tab_name`, the others do not), the extra parameter needs a
  default.
- **Sync or async:** check how the step calls your function. When it
  goes through the emitted `_call_impl`, both plain and `async def`
  functions work. When it calls the function directly, write a plain
  function, because a coroutine would come back unawaited.
- **Return value:** a single-type `output` (`output: raw_content str`)
  is returned bare, and downstream receives exactly that value. A map
  `output` is a dict with those keys. Pydantic models are fine because
  `_serialize` dumps them, but plain dicts and lists are simpler.
- **IR-adjacent source:** if `extracted/<module>.py` exists beside the
  `pipeline.yaml`, emission copies it into the runtime dir verbatim.
  Write there and re-emit when that directory exists.
- **Judges on temporal's legacy path:** a `signature: 'path.py:Class'`
  judge without `signature_spec` imports a class you write. A
  `signature_spec` judge is emitted working on every runtime.
- **Run:** `uvx --from 'rote-cli[dbos]' rote run <dir> --input ... --signal ...`
  for dbos, which uses a local SQLite file by default.
  `'rote-cli[temporal]'` for temporal, which starts a local Temporal dev
  server itself. The python runtime refuses pipelines with HITL gates at
  emit time.

## TypeScript runtimes (cloudflare, dbos-ts, inngest)

- **Stubs:** `src/extracted/<node_id>.ts`, one file per node, even when
  nodes share a Python `impl`. Each exports a camelCase function
  (`parse_dock_pending` exports `parseDockPending`) that the workflow
  imports by that name.
- **Call shape:** one object argument whose keys are the node's
  `inputs:` parameter names. The stub types it `_input: unknown`.
  Replace that with a real interface for the input.
- **Return type:** the stub declares `Promise<never>` because it always
  throws. Replace it with a concrete type. On Cloudflare the return
  value must satisfy `Rpc.Serializable`, so return plain objects,
  arrays, strings, and numbers, never `Record<string, unknown>` or class
  instances. The workflow reads fields as
  `(<id>_result as Record<string, unknown>)["field"]`, so any concrete
  object type compiles.
- **Constants are not passed.** The workflow sends only the node's
  `inputs:`. The stub's doc comment lists the node's constants under
  "Constants from the source skill"; write them into the file as named
  constants. One file serves one node, so there is no ambiguity.
- **Agent-loop modules are not stubs.** `src/extracted/<loop node id>.ts`
  is generated working code. Leave it alone.
- **Python code does not carry over.** The compiler's
  `extracted/*.py` beside the IR is not used by TS runtimes. Port the
  logic.
- **Check:** `npm install && npx tsc --noEmit` in the emitted dir.
- **Run:** `rote run <dir>` drives `wrangler dev --local` for
  cloudflare, a managed `inngest-cli dev` for inngest, and Postgres
  (falling back to a throwaway Docker container) for dbos-ts. Each needs
  Node, and dbos-ts also needs Docker or a reachable Postgres.

## Stub markers

`rote emit --json` reports a file in `unimplemented_stubs` while it
still contains any of these strings. Make sure your implementation does
not contain them either, even in a comment:

- `NotImplementedError` (Python)
- `stub not implemented` (TypeScript)
- `requires an agent runtime` (TypeScript agent loops on older emits)
