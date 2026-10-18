# Where stubs live, per runtime

Each runtime puts user code in a different place and calls it with a
different shape. Confirm against the generated workflow file every time.
That file is the contract the emitter actually wrote, and this page can
drift from it.

## Python runtimes

### dbos, python

- **Stubs:** `extracted/<module>.py`, one module per `impl` path. Several
  nodes can share one function (an IR `impl` of
  `extracted/google_drive.py:fetch_file_content` on three nodes is one
  function called three times).
- **Call shape:** `main.py` calls `func(**payload)`. The payload keys are
  the node's `inputs:` parameter names, plus the node's constants. Read
  the step body in `main.py` for the exact keys. Give the function
  explicit keyword parameters rather than `**payload`, so a contract
  change fails loudly.
- **Sync, not async.** DBOS steps and the python script call the
  function directly.
- **Return value:** a single-type `output` (`output: raw_content str`)
  is returned bare, and downstream receives exactly that value. A map
  `output` is a dict with those keys. Pydantic models are fine because
  `_serialize` dumps them, but plain dicts and lists are simpler.
- **IR-adjacent source:** if `extracted/<module>.py` exists beside the
  `pipeline.yaml`, emission copies it into the runtime dir verbatim.
  Write there and re-emit when that directory exists.
- **Run:** `uvx --from 'rote-cli[dbos]' rote run <dir> --input ... --signal ...`.
  DBOS uses a local SQLite file by default, so no infrastructure is
  needed. The python runtime refuses pipelines with HITL gates at emit
  time.

### temporal

- **Stubs:** none are emitted. `activities.py` imports
  `from <package>.extracted.<module> import <func>`, where `<package>`
  is the adapter's configured module path (`expected` by default).
  Create that package on the import path (`<package>/__init__.py`,
  `<package>/extracted/__init__.py`, `<package>/extracted/<module>.py`)
  next to the emitted files.
- **Async.** Activities `await func(**payload)`, so implementations must
  be `async def`. Use an async HTTP client, or run blocking SDK calls in
  `asyncio.to_thread`.
- **Return value:** the same rules as dbos.
- **Judges on the legacy path:** a `signature: 'path.py:Class'` judge
  without `signature_spec` imports a class you write. A `signature_spec`
  judge is emitted working.
- **Run:** `uvx --from 'rote-cli[temporal]' rote run <dir> ...`. It
  starts a local Temporal dev server itself.

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
- **Constants:** read them off the input like the other keys, and check
  the call site in the workflow file.
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
