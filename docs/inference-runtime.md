# Inference Runtime Decision Record

How an emitted pipeline's `llm_judge` steps get their inference, who
pays for it, and how that choice is made.

This is the sibling of [`agent-runtime.md`](agent-runtime.md), which
records the same decision one layer up — for the compiler *agent*.
Read that first. Everything here is deliberately shaped to match it,
because rote should have exactly one way of answering "which model
provider, billed to whom".

---

## Context

Three layers of rote invoke a model, and until now only two of them
had an answer to "whose money".

| Layer | What runs | Selection | Billing |
| --- | --- | --- | --- |
| Compiler agent | `rote compile` | `--agent` / `agent:` | subscription (`claude`, `codex`) or key (`api`) |
| Skill baseline | `rote run <skill>`, `rote eval --run` | fixed: `claude -p` | subscription |
| **Emitted judge** | the compiled pipeline | **none — API key only** | **user's key, always** |

The gap has a real cost. A user with a Claude Max subscription can
compile a skill for free and then cannot run the result without
putting a paid API key somewhere, which is a strange cliff to fall off
at the exact moment the product is supposed to be paying off. And a
user logged into rote cloud has a billing relationship with us that
emitted judges cannot use.

## Decision

Emitted judges select an **inference provider** at runtime, from three
— but there are only **two transports**, and that is the point:

| Provider | Transport | Billing | Runtimes |
| --- | --- | --- | --- |
| `claude-cli` | subprocess to `claude -p` | Claude subscription | Python, **local only** |
| `api` | vendor SDK at a resolved endpoint | user's API key | all |
| `rote-cloud` | *the same SDK call*, different endpoint + token | tenant account | all |

`api` and `rote-cloud` are not two implementations. They differ only in
`(base_url, auth header, model id)` — which is exactly what the
existing `ROTE_BASE_URL_<ID>` / `ROTE_MODEL_<ID>` knobs already carry,
and was proven empirically: pointing a judge at a Cloudflare AI Gateway
with Unified Billing (Cloudflare holds the provider credential, bills
the account) required **zero code changes**, only environment.

So a provider is a **resolver**, not a client:

```
resolve(provider) -> (transport, base_url | None, auth env, model)
```

Two transports get written and tested — `sdk` and `cli`. A fourth
billing lane later is a resolver entry, not a new code path.

Selection mirrors the driver layer exactly:

* a `name` and a `provider_availability() -> (bool, reason)`;
* an auto-detect order that **prefers the subscription** and falls
  through to paid paths;
* explicit user choice always beats auto-detect;
* no silent fallback — when nothing is available the error lists the
  setup step for each provider, and an explicitly chosen provider that
  cannot serve the judge is an **error**, never a downgrade. Falling
  back from `claude-cli` to `api` would move the bill from a flat
  subscription onto a metered key without anyone saying so.

Surfaced as `--inference` on the commands that run a pipeline, an
`inference:` key in `rote.yaml` / the user config, and
`ROTE_INFERENCE` (globally) or `ROTE_INFERENCE_<NODE_ID>` (per node),
resolving in the layered order `rote.config` already implements:
flag > `ROTE_*` env > project `rote.yaml` > user config > built-in.

The CLI's only job is to **resolve and export** `ROTE_INFERENCE`.
That env var is the emitted app's own interface — a generated pipeline
reads it whether rote launched it or a user did — so there is no
private channel from CLI to app, and `--inference` is a convenience
over an interface that stands on its own.

`--backend` is **not** reused: it already means "how does an
`external_call` node reach its tool" (`mcp` | `api`). Overloading it
would make `--backend api --inference api` read as one setting.

### One thing the resolver must know besides the provider

A judge with an explicit endpoint (`signature_spec.base_url` or
`ROTE_BASE_URL_<NODE_ID>`) rules `claude-cli` **out** of auto-detect.
The proven Cloudflare AI Gateway config is exactly that shape, and a
laptop that happens to have `claude` on PATH would otherwise silently
ignore the endpoint the operator deliberately pointed the judge at.

## The load-bearing boundary: provider is not IR

`signature_spec.client` (`anthropic` | `openai` | `workers-ai`) stays
in the IR. The provider does not, ever.

The distinction is *wire format* versus *who pays and by what
transport*. The former is behavior — it determines whether the judge
gets a forced tool call or a `json_schema` response, which changes
what the pipeline does. The latter is a deployment fact about one
person's machine on one day.

This is invariant #1 (the IR is runtime-agnostic and behavior-only)
applied to money, and the same reasoning that keeps the eval sidecar
out of `pipeline.yaml`. A `pipeline.yaml` is shared, committed, and
sometimes third-party — a billing decision encoded there would travel
to people it does not belong to. **A pipeline.yaml must never say who
pays.**

Practical consequence: the same emitted artifact runs on a
subscription locally, on a key in CI, and on rote cloud in
production, with no re-emit and no diff.

## Where the provider code lives

Emitted apps cannot import `rote` — they are standalone projects with
their own dependency sets. So the provider implementations ship
**verbatim** into the emitted tree as `signatures/_rote_inference.py`,
sourced from `rote.inference._runtime_helper`.

This is the established pattern, not a new one:
`extracted/_rote_mcp.py` is the verbatim source of
`rote.mcp._runtime_helper`, and the same rule applies here — **never
hand-edit the emitted copy; fix the module and re-emit.** The
equivalent regression test (`tests/test_mcp_client.py` for MCP) is the
model for this helper's tests.

Each emitted judge shrinks to: interpolate the prompt, call
`call_judge(...)`, validate the result against the generated Pydantic
model. Vendor call code stops being duplicated per node.

### One definition, copied mechanically — not duplicated

`build_subscription_env()` is the subscription rule: scrub
`ANTHROPIC_API_KEY` and `ANTHROPIC_AUTH_TOKEN` (in `claude -p` they
always beat an active OAuth session, defeating the entire point),
preserve `CLAUDE_CODE_OAUTH_TOKEN` for CI, silence non-interactive
animations. Today it lives in
`rote.compiler.drivers.claude` and is imported by the baseline runner
and the empirical eval runner.

It **moves into `rote.inference._runtime_helper`**, and
`drivers/claude.py` imports it back from there. The emitted app needs
that exact rule for its `claude-cli` provider, the repo needs it for
the compiler, and neither gets its own copy — there is one definition
and one byte-identical build artifact, asserted the same way
`_rote_mcp.py` is:

```python
emitted = (out / "signatures" / "_rote_inference.py").read_text()
assert emitted == Path(_runtime_helper.__file__).read_text()
```

Constraint this places on the module: **stdlib only**, no imports from
`rote`. It executes inside emitted apps where rote is not installed.
That is the same constraint `rote.mcp._runtime_helper` already lives
under, and it is what makes the copy safe rather than a fork.

## The cloud runs the CLI; the cloud route stays a proxy

rote-cloud's compilation container already installs
`rote-cli[api,openai-api]` and runs the OSS compiler — the platform
does not reimplement compilation, it *runs rote*. Inference keeps that
rule, from the other end.

The `rote-cloud` provider sends the **vendor's own wire format** to
`<cloud>/v1/inference/<vendor>` with a tenant token. The server
authenticates, meters, and forwards to a provider it holds credentials
for (the same Unified-Billing gateway shape validated locally). It
does not interpolate prompts, does not own schemas, does not know what
a judge is.

That keeps the split clean in both directions: **anything that
understands rote's IR runs the CLI; anything that only moves tokens is
a proxy.** A server that parsed `signature_spec` would be a second
implementation of the judge, in another language, drifting from the
first — the precise thing we are avoiding by shipping the helper
verbatim instead of hand-writing it per runtime.

**Status:** the client half ships here and is proven; the server route
does not exist yet (a real call returns 404 from `app.roteskills.com`).
Until rote-cloud serves `/v1/inference/<vendor>`, the `rote-cloud`
provider resolves and fails cleanly rather than silently — which is the
correct behavior for a lane whose backend is not deployed.

Corollary for TypeScript: TS runtimes have no subprocess transport, so
they need **no provider machinery at all** — only the endpoint/auth
resolution they already have. There is no TS port of the judge logic
to keep in sync.

## `claude-cli` is a local provider

It is available to `rote run` and to a self-hosted Python pipeline on a
developer machine.

Three reasons it cannot be more than that, in order of weight:

1. A Claude subscription credential is personal. Baking it into a
   server process makes one human's login a production dependency.
2. `agent-runtime.md` already records that Anthropic's terms forbid
   third-party agents on the Agent SDK from using claude.ai login
   credentials without approval. Spawning the CLI as a coding agent is
   what the CLI is for; wiring it in as a production inference
   endpoint for someone else's workload is a different thing, and we
   should not be the tool that blurs it.
3. Cloudflare Workers has no subprocess at all, so the provider cannot
   be universal regardless.

### The refusal belongs to `deploy`, not `emit`

An earlier draft of this document put it at emit time, by analogy with
the Python adapter refusing `hitl_gate` nodes. That analogy is wrong:
`hitl_gate` is a property of the *pipeline*, so the adapter can see it.
The provider deliberately is not — that is the whole point of the
section above — so `rote emit` genuinely cannot know whether the
directory it is writing will be run on a laptop or pushed to a
container.

`rote deploy` can. It refuses when the resolved provider is
`claude-cli`, names the config layer that set it, and points at the
one-line fix — explicitly noting that the *emitted code needs no
change*, because that is the property being protected.

## Fidelity: equal, which was not the expectation

The draft of this document assumed `claude-cli` would be the weak
provider — schema in the prompt, validate on the way out, retry on
prose. Testing the CLI disproved that: `claude -p --json-schema <s>
--output-format json` performs a **forced tool call** (the envelope
reports `stop_reason: "tool_use"`) and returns the validated object in
a `structured_output` field. Same structural guarantee as the SDK path.

Two things still differ, and both are cost, not correctness:

* **Latency.** A CLI spawn is seconds of process startup the SDK path
  does not pay.
* **Prompt overhead.** Claude Code sends its full coding-agent system
  prompt by default — measured at ~37k cache-creation tokens and 11.5s
  for a one-sentence judge. Replacing it (`--system-prompt`), dropping
  tools (`--tools` with no arguments) and ignoring the user's settings
  files (`--setting-sources ""`) brings the same call to ~740 tokens
  and ~3.5s. The emitted helper always passes all three; do not
  "simplify" them away.

The provider is still stamped into every `$ROTE_USAGE_LOG` record and
must reach `rote eval`'s corpus, because the same token count costs
differently per lane — a subscription trial is not priced against a
key trial. That is a *billing* distinction now, not a quality one.

## Auto-detect behavior

One order, everywhere, cheapest-to-the-user first:

1. `claude` CLI on PATH, judge is anthropic-client, and no explicit
   endpoint is configured → `claude-cli`. The user already paid for it.
2. Else an API key for the node's `client` is set → `api`. Their own
   key, at provider rates.
3. Else logged into rote cloud → `rote-cloud`.
4. Else fail, listing the setup step for all three.

The draft had a second, inverted order for deployed pipelines. It is
not needed: a deployed image has no `claude` binary, so candidate 1
drops out on its own. One code path, no "am I deployed?" question that
nothing can answer reliably anyway.

## Naming to keep straight

| Name | Means |
| --- | --- |
| `--agent` / `ROTE_AGENT` | which **compiler driver** runs the agent loop |
| `--model` / `ROTE_MODEL` | the **compiler's** model |
| `--inference` / `ROTE_INFERENCE` | which **judge provider** serves emitted judges |
| `ROTE_MODEL_<NODE_ID>` | one **judge's** model |
| `--backend` | how an `external_call` reaches its tool (`mcp` \| `api`) |

`ROTE_MODEL` and `ROTE_MODEL_<NODE_ID>` are adjacent and mean
different things. That is survivable because the suffixed form is
always node-scoped, but do not add a bare `ROTE_TEMPERATURE` or
`ROTE_BASE_URL` — the unsuffixed namespace belongs to the compiler.

## References

* [`agent-runtime.md`](agent-runtime.md) — the same decision for the
  compiler agent, and the source of this document's shape
* [`mcp-client.md`](mcp-client.md) — the verbatim-shipped-helper
  pattern this reuses
* `rote.config` — the layered resolution every selection flag uses
