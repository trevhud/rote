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
- **Call shape:** each node's step calls
  `_call_impl(func, {**payload, "<constant>": <value>, ...})`, so the
  function receives the node's `inputs:` parameter names plus its IR
  `constants`, all as keyword arguments. The constants are merged last,
  in the step itself, so every caller gets them: the DAG, each fan_out
  element, and an agent loop calling the node as a tool. Read that call
  in `main.py` (dbos, python) or `activities.py` (temporal) for the exact
  keys per node. Give the function explicit keyword parameters rather
  than `**payload`, so a contract change fails loudly, and let the
  constants it receives drive the logic. When nodes sharing a function
  send different constant sets (one sends `tab_name`, the others do
  not), the extra parameter needs a default.
- **Sync or async:** either works on all three. `_call_impl` awaits an
  `async def` function (or a plain one that returns a coroutine) on dbos
  and python, and on temporal it runs a plain function in a worker
  thread so it cannot block the event loop.
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
  "Constants from the source skill". Write them into the file as named
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

## What counts as a stub

`rote emit --json` lists a module in `unimplemented_stubs`, and `rote run`
refuses to start, while any function in it is still a stub. The check
reads the file off disk, so a module drops off the list as soon as its
last stub function is filled in. It looks only at files the emit
manifest lists under an `extracted/` directory, skipping `__init__`.

Every stub rote emits raises with the sentinel `[rote stub]` in its error
message. On top of that:

- **Python:** a function is a stub when a `raise` passes a string
  carrying `[rote stub]`, or when its whole body after the docstring is
  one `raise NotImplementedError` (the shape a compile agent writes).
  Class methods count. An `@abstractmethod` does not, and neither does a
  real function that raises `NotImplementedError` on one unsupported
  branch. The sentinel in a comment or docstring does not count. A file
  that does not parse is not reported here, so check `contract_findings`
  and run it.
- **TypeScript:** a file is a stub while `[rote stub]` appears anywhere
  outside a comment, even in a string that is never thrown. A comment
  mentioning it does not count. A `throw` without the sentinel is not
  detected, which is one more reason never to leave a placeholder.

Each block below is checked against rote's detector by the test suite.

```python stub
def resolve_taxonomy_ids(levels: list[str]) -> dict:
    """Resolve ZoomInfo IDs."""
    raise NotImplementedError
```

```python stub
class HubSpotClient:
    def upsert(self, contacts: list[dict]) -> dict:
        raise RuntimeError("[rote stub] hubspot.upsert: implement against the vendor API")
```

```python not-stub
from abc import ABC, abstractmethod


class Source(ABC):
    @abstractmethod
    def fetch(self) -> str:
        raise NotImplementedError


def parse(kind: str, text: str) -> dict:
    """Replaced the [rote stub]: parses CSV exports."""
    if kind == "csv":
        return {"rows": text.splitlines()}
    raise NotImplementedError(f"unsupported export kind {kind!r}")
```

```ts stub
export async function taxonomyLookup(_input: unknown): Promise<never> {
    throw new Error("[rote stub] pure_function taxonomy_lookup: stub not implemented");
}
```

```ts not-stub
// Was a [rote stub]; now calls the ZoomInfo lookup API.
export async function taxonomyLookup(input: { levels: string[] }): Promise<string[]> {
    return input.levels.map((level) => level.toUpperCase());
}
```

When the run will not reach a remaining stub (it sits on a branch this
input does not take), `rote run --allow-stubs` skips the refusal. The
stub still raises if the run does reach it.
