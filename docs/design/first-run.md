# First-run onboarding: architecture

Status: foundation landed (#241); every client source landed (PR 2); `rote mcp import`
landed (PR 3, below). `rote setup` and its other steps are follow-up PRs built to the
interfaces below. Product design: "rote first-run design"
(approved 2026-10-22, all nine recommendations taken).

The rule everything here serves: **every side effect is an Action in a Plan,
shown and confirmed before it runs, and interactivity is decided once, before
the first question.** A file write, a server contacted, a browser opened, or
money spent never happens inline in a command. It happens because a person
said yes to a plan that listed it, or because a script passed `--yes` (plus
`--force` to replace a file, plus the spend's own flag to spend money).

## What we took from which CLI

| Pattern | Source | Here |
| --- | --- | --- |
| One IO object decides `CanPrompt()` once; nothing else calls `isatty` | gh [`pkg/iostreams/iostreams.go`](https://github.com/cli/cli/blob/HEAD/pkg/iostreams/iostreams.go), [`internal/ghcmd/cmd.go`](https://github.com/cli/cli/blob/HEAD/internal/ghcmd/cmd.go) | `rote.interact.mode.Mode.resolve` |
| `GH_PROMPT_DISABLED` | gh | `ROTE_PROMPT_DISABLED` |
| A `Prompter` interface, stubbed per question in tests | gh [`internal/prompter/prompter.go`](https://github.com/cli/cli/blob/HEAD/internal/prompter/prompter.go), [`pr/create/create_test.go`](https://github.com/cli/cli/blob/HEAD/pkg/cmd/pr/create/create_test.go) | `rote.interact.prompter.Prompter`, `ScriptedPrompter` |
| Non-interactive errors name the flag to pass | gh `cmdutil.FlagErrorf("must provide --title ...")` in [`pr/create/create.go`](https://github.com/cli/cli/blob/HEAD/pkg/cmd/pr/create/create.go) | `PromptRequired`, exit 2 |
| Dependencies injected through one struct of fields | gh [`pkg/cmdutil/factory.go`](https://github.com/cli/cli/blob/HEAD/pkg/cmdutil/factory.go) | `ClientContext`, `SetupContext` |
| Agent detection from env, with an injectable lookup | gh [`internal/agents/detect.go`](https://github.com/cli/cli/blob/HEAD/internal/agents/detect.go), [`@vercel/detect-agent`](https://github.com/vercel/vercel/blob/HEAD/packages/detect-agent/src/index.ts) | `rote.interact.agents` |
| A detected agent means non-interactive | Vercel [`util/client.ts`](https://github.com/vercel/vercel/blob/HEAD/packages/cli/src/util/client.ts) (`nonInteractive = opts.nonInteractive ?? this.isAgent`) | `Mode.agent` |
| Prompts render to stderr, so stdout stays the result | Vercel `client.input` | `QuestionaryPrompter` (`create_output(stdout=sys.stderr)`) |
| `action_required` payload with `missing` and `next: [{command}]` | Vercel [`util/agent-output.ts`](https://github.com/vercel/vercel/blob/HEAD/packages/cli/src/util/agent-output.ts) | `rote.interact.report.action_required` |
| `--agent auto\|yes\|no` | Supabase [`internal/utils/agent.go`](https://github.com/supabase/cli/blob/HEAD/apps/cli-go/internal/utils/agent.go) | `--agent-mode` (see Deviations) |
| An unrecognized answer on a pipe is never consent | Supabase TS [`prompt-yes-no.ts`](https://github.com/supabase/cli/blob/HEAD/apps/cli/src/command-internal/prompt-yes-no.ts) (its Go version's default-on-timeout is the anti-pattern) | `NonInteractivePrompter` never reads stdin |
| Plan printed inside the confirmation; side effects documented per command | Supabase [`db-push-core.ts`](https://github.com/supabase/cli/blob/HEAD/apps/cli/src/command-internal/db-push-core.ts), [`SIDE_EFFECTS.md`](https://github.com/supabase/cli/blob/HEAD/apps/cli/src/commands/db/push/SIDE_EFFECTS.md) | `Plan` + `Effect` records |
| Login as a two-step resumable flow (`--non-interactive`, then `--complete`) | Stripe [`pkg/login/oauth_pending.go`](https://github.com/stripe/stripe-cli/blob/HEAD/pkg/login/oauth_pending.go), [`cmd/login.go`](https://github.com/stripe/stripe-cli/blob/HEAD/pkg/cmd/login.go) | `rote mcp login --no-wait` / `--complete` (PR 4, `rote.mcp.pending_login`) |
| Output channels as capabilities, not branches | uv [`printer.rs`](https://github.com/astral-sh/uv/blob/HEAD/crates/uv-command-support/src/printer.rs) | `rote.interact.output.Output` (`say` to stderr, `result` to stdout) |
| Steps that see earlier answers; cancel as a value | clack [`group.ts`](https://github.com/bombshell-dev/clack/blob/HEAD/packages/prompts/src/group.ts), [`core/utils`](https://github.com/bombshell-dev/clack/blob/HEAD/packages/core/src/utils/index.ts) | `SetupContext.results`; `PromptCancelled` |
| Plan/apply split; a trivial plan never prompts; JSON mode refuses rather than auto-approving | Terraform [`backend_apply.go`](https://github.com/hashicorp/terraform/blob/HEAD/internal/backend/local/backend_apply.go), [`arguments/apply.go`](https://github.com/hashicorp/terraform/blob/HEAD/internal/command/arguments/apply.go) | `confirm_plan`: empty plan needs nothing, `--json` without `--yes` stops |

Anti-patterns we avoid: Vercel's module-level `output` singleton (not
injectable), Vercel and Stripe re-checking `isTTY` at each decision point,
Supabase Go treating a pipe's silence as the default answer, and clack having
no non-interactive mode at all (the caller must gate it, which is what
`prompter_for` is).

## Module layout

```mermaid
flowchart LR
  subgraph interact["rote.interact (terminal layer)"]
    agents[agents.py<br/>env → Detected]
    mode[mode.py<br/>Mode.resolve]
    prompter[prompter.py<br/>Prompter + 3 impls]
    plan[plan.py<br/>Effect, Action, Plan]
    output[output.py<br/>Output]
    report[report.py<br/>action_required]
  end
  subgraph clients["rote.clients (read-only sources)"]
    base[base.py<br/>ClientSource, ImportCandidate]
    normalize[normalize.py<br/>names, placeholders, headers, URLs]
    claude[claude_code.py]
    common[common.py<br/>reader, candidate, rank, merge]
    more[codex.py, cursor.py, vscode.py, ...<br/>one module per client]
  end
  setup[rote.setup<br/>Step, run_steps]
  init[init_wizard.py<br/>rote init]
  cli[cli.py<br/>_cmd_init, later _cmd_setup, _cmd_mcp_import, _cmd_try]

  agents --> mode --> prompter
  plan --> output
  prompter --> report
  plan --> report
  normalize --> common --> claude --> base
  common --> more --> base
  init --> prompter & plan & output
  setup --> prompter & plan & output
  cli --> mode & init & setup & report
  setup -. steps use .-> clients
```

`rote.interact` knows nothing about MCP or skills, and `rote.clients` knows
nothing about terminals. `rote.setup` and the commands are where they meet.

## Mode: decided once

`Mode.resolve(json=, yes=, agent_mode=, env=, stdin=, stdout=, stderr=)` reads
the terminal and environment one time. Every argument is injectable.

```mermaid
flowchart TD
  A[command entry] --> B[Mode.resolve]
  B --> C{--json?}
  C -- yes --> N[NonInteractivePrompter]
  C -- no --> D{--yes?}
  D -- yes --> NY[NonInteractivePrompter<br/>accept_defaults]
  D -- no --> E{ROTE_PROMPT_DISABLED?}
  E -- set --> N
  E -- no --> F{agent caller?<br/>--agent-mode, ROTE_AGENT_MODE, env signals}
  F -- yes --> N
  F -- no --> G{stdin and stderr TTY?}
  G -- no --> N
  G -- yes --> Q[QuestionaryPrompter<br/>drawn on stderr]
```

Agent signals have a **role**. A *caller* (`CLAUDE_CODE_CHILD_SESSION`,
`CODEX_THREAD_ID`, `GEMINI_CLI`, `AI_AGENT`, ...) means an agent process
started rote: no prompts. A *host* (`CLAUDECODE`, `CURSOR_TRACE_ID`,
`ZED_TERM`, and `REPL_ID` and `GOOSE_PROVIDER`, both set in a person's own
shell) only says which client's terminal this is. Claude Code sets
`CLAUDECODE=1` in IDE terminals too, where a person types. Hosts rank the
import sources and never turn prompts off. `AI_AGENT` is matched by prefix:
Claude Code 2.1.302 sets `AI_AGENT=claude-code_2-1-302_agent`.

stdout is not part of the decision: prompts draw on stderr, so
`rote setup | tee log` still asks.

**Why "decided before the first prompt" is a hard rule:** prompt_toolkit reads
a piped stdin as keystrokes. `printf '\r' | <questionary prompt>` returns the
default with only a warning on stderr, an empty pipe raises `EOFError`, and
`.ask()` prints "Cancelled by user" to *stdout* on Ctrl-C. So the CLI never
constructs a `QuestionaryPrompter` without a terminal, uses `unsafe_ask()`,
and `tests/test_init.py::test_piped_answers_never_reach_a_prompt` pipes
`\r\r\r...y\r` into a real `rote init` subprocess and requires exit 2 with
nothing written.

## Prompter

```python
class Prompter(Protocol):
    interactive: bool  # property
    def select(self, message, choices: Sequence[Choice[T]], *, flag, default=None) -> T
    def checkbox(self, message, choices, *, flag, default: Sequence[T] = ()) -> list[T]
    def confirm(self, message, *, flag, default: bool) -> bool
    def text(self, message, *, flag, default=None, validate=None) -> str
```

Every question names its flag. The command checks the flag first and asks
only when it is absent (gh's shape):

```python
runtime = flags.runtime or prompter.select("Which runtime?", RUNTIMES, flag="--runtime", default="dbos")
```

| Implementation | When | Behavior |
| --- | --- | --- |
| `QuestionaryPrompter` | a person at a terminal | questionary 2.1.1 on prompt_toolkit 3.0.53, output on stderr, `unsafe_ask()`, Ctrl-C/Ctrl-D raise `PromptCancelled`. `input=`/`output=` injectable (tests use `create_pipe_input` + `DummyOutput`). |
| `NonInteractivePrompter(why, accept_defaults)` | everything else | never reads stdin. With `--yes` returns each default. A question with no default, or any question without `--yes`, raises `PromptRequired(question, flag, why, choices)`. |
| `ScriptedPrompter(answers)` | tests | each `Answer(kind, flag, value)` must match the question it answers, so a test fails where the flow diverged. `DEFAULT` takes the question's default; `interactive_` simulates either side. |

`Choice(value, label, hint, disabled)`: a disabled choice is shown with its
reason (a stdio server, a URL with a query string) and can never be picked or
be a default.

`prompter.interactive` exists so a command can skip what only a person can do
(open a browser, offer a login). It never picks a different flow.

## Plan and Action

```python
Effect = FileWrite(path, summary, existed, replaces, file_mode)
       | Contact(host, purpose)        # a bare host: never a URL
       | Launch(program, purpose, paid)  # a browser, `claude -p`
       | Spend(low_usd, high_usd, payer, flag)

Action(id, title, effects: tuple[Effect, ...], run: Callable[[], object], command: str | None)
Plan(title, actions, notes)
Outcome(action_id, status, error, result)   # result = what run() returned
```

- An `Action` with no effects is refused: if nothing needs previewing it is
  not a side effect and does not belong in a plan.
- The code that applies an action is attached to the record that previews it,
  so the confirmation is about the real effects.
- `Action.command` is the exact standalone command that does the same thing
  (`rote mcp login figma`), which is what an agent runs later.
- **A write is re-checked at apply, against the caller's own read.**
  `write_file_action(read=snapshot(path), ...)` takes the `Snapshot` the
  caller built its content from (a merge parses `read.data`), so the content
  and the stale check describe the same bytes: an edit landing between the
  caller's read and the action is caught too (review6). It records the
  file's state from that read (absent, or the SHA-256 of its bytes), and its
  `run()` refuses with `StaleFile` ("changed since the plan was made; nothing
  written, re-run") when the file no longer matches. The check runs after the
  new content is staged and right before the rename, so the window left is the
  rename, not the minutes a person spends at "Apply this plan?". Without it, a
  plan confirmed as a create (so `--yes` alone sufficed) would silently replace
  a file that appeared meanwhile.
- **A paid launch carries its spend.** `Launch(paid=True)` (a compile, a skill
  run through `claude -p`) is refused by `Action` unless the same action holds
  a `Spend`, whose flag is what keeps `--yes` from ever starting it.

```mermaid
sequenceDiagram
  participant C as command
  participant P as Prompter
  participant O as Output
  C->>P: questions (each with its flag)
  C->>C: build Plan (no side effects)
  C->>O: plan(plan) to stderr, or result(plan.to_dict()) under --json
  C->>P: confirm_plan(plan, yes=, granted=)
  alt a person
    P-->>C: yes/no (default no if it replaces a file or spends)
  else nobody at the terminal
    C-->>C: needs --yes, plus --force / the spend's flag, else ConfirmationRequired
  end
  C->>C: apply_plan(plan): in order, stop at first failure
  C->>O: outcomes (done / failed / not-run)
```

Consent rules, in `confirm_plan`:

| Effect | A person | Nobody at the terminal |
| --- | --- | --- |
| create or update a file, contact a host, launch a program | asked, default yes | `--yes` |
| replace an existing file | asked, default no | `--yes --force` |
| spend money | asked, default no | `--yes` plus the `Spend.flag` (`--live`). `--yes` alone never spends |

`atomic_write_text` (temp file in the same dir, `os.replace`, mode set before
the rename) is the one way an action writes a file. `rote.config.write_config`
now uses it too.

## `--json`: the same Plan, rendered once

rote's `--json` rule (one object on stdout) holds. A command that applied
prints `{"ok", "applied": true, "plan": plan.to_dict(), "outcomes": [...]}`.
A command that stopped prints an `action_required` object and exits 2:

```json
{
  "ok": false,
  "status": "action_required",
  "reason": "missing_answer",
  "message": "Which runtime should rote emit by default: pass --runtime <dbos|temporal|...> (...)",
  "missing": ["--runtime"],
  "choices": ["dbos", "temporal", "cloudflare", "dbos-ts", "inngest", "python"],
  "next": [{"command": "rote init --deploy none --json --runtime <dbos|temporal|...>"}]
}
```

`reason: "confirmation_required"` carries `"plan": plan.to_dict()` instead,
with `"needs": ["--yes", "--force", ...]`, so the agent can show the user what
would happen before it re-runs `next`. `next` is the user's own argv plus the
missing flags (`args.raw_argv`, set in `main`).

## Output

`Output(mode, stdout=None, stderr=None)`: `say`, `warn`, `error`, `plan` go to
stderr through rich (color only when stderr is a terminal and `NO_COLOR` is
unset); `result` writes the one JSON object to stdout under `--json` and
nothing otherwise. Streams resolve per write, so `cli._run_json_command`'s
redirection and pytest's capture see everything. Text from other apps' configs
is printed as rich `Text` with `markup=False`: a server named `[link=...]`
cannot restyle or relink a line.

## Client sources

```python
class ClientSource(Protocol):
    id: str       # "claude-code", the --from value and the agents.py client id
    label: str
    def config_paths(self, ctx: ClientContext) -> tuple[Path, ...]
    def detect(self, ctx: ClientContext) -> bool
    def read(self, ctx: ClientContext) -> SourceReading   # never raises for a bad file

ClientContext(home, cwd, env, system_dir, etc_dir, platform)  # injected; .current() for real runs
SourceReading(client, candidates, files_read, problems)
ImportCandidate(client, name, original_name, scope, origin, transport, url (never shown),
                headers: tuple[HeaderTemplate], scopes, withheld, blocked, inactive, plugin)
```

A source only reads files. It never runs the client (`claude mcp list`
health-checks every server, work servers included), never contacts a server,
never writes, and never resolves a placeholder. `ClientContext` is the only
door to the machine, so tests point a source at fixture files and a source
cannot read the real home by accident; `ROTE_CLIENT_HOME_DIR` (in
`rote.isolation.STATE_PATHS`, so conftest and `scripts/isolated-rote` move it)
does the same for `ClientContext.current()`, dropping `CLAUDE_CONFIG_DIR`,
`CODEX_HOME` and `XDG_CONFIG_HOME` so they cannot lead back to the real files.

`rote.clients.normalize` holds the rules every source applies the same way:

- **Names** become identifiers (`acme-crm-prod` to `acme_crm_prod`, `9lives`
  to `_9lives`); the original is recorded.
- **Placeholders**: each client's syntax is a `PlaceholderSyntax`; references
  are rewritten to `${VAR}` and never resolved. A `${VAR:-default}` default is
  dropped and reported (it may be the secret). Client-internal variables
  (`${CLAUDE_PLUGIN_ROOT}`) and inputs (VS Code `${input:id}`) are unsupported.
- **Headers** are copied only when the value is references plus at most an
  auth scheme word (`Bearer ${GITHUB_TOKEN}`). Anything else is a literal,
  withheld with the `rote mcp add <name> <url> --header` command that sets it.
- **URLs** are copied only as plain http(s): no userinfo, no query string
  (API keys travel there), no placeholder.
- **Display**: `ImportCandidate.to_dict()` has the host, never the URL; header
  names and templates, never values; and `withheld`, each item with its reason.
  `headersHelper` and `oauth.clientId` (issued to Claude Code) are withheld.
- **A credential-shaped URL path** (`url_secret_risk`): a path segment of 20+
  characters with Shannon entropy of 3.5+ bits per character in which one
  part, split on `-`, `_` and `.`, is 8+ characters mixing letters and
  digits (a key's random body: `sk-ak-<body>`, `ghp_<body>`) or three or
  more parts that each mix them (a grouped key, `A7F2K-9QXM3-PL8RT-...`), or a 32+ hex
  run (UUID dashes removed), sets `ImportCandidate.risk`. The part rule keeps
  slugs such as `my-awesome-server-2024` and
  `salesforce-opportunity-management-v2` importable: their letters and
  digits mix only across words. Decided by the string's form, never a word list.
  Hosted servers put keys there (Zapier's `/mcp/<key>/sse`). Such a candidate
  is importable but not `preselectable`: no default and no `--yes` picks it;
  `rote mcp import` needs a per-server opt-in that names the risk
  (`--allow-url-secret <name>`). The reason never quotes the segment.
- `is_work_host(host, work_domains)`: suffix match on whole labels.
- **Collisions** (`rote.clients.collisions.find_collisions(candidates,
  registered)`): two active, importable candidates that map to one identifier
  with different URLs (`acme-crm`, `acme_crm`), or a name already registered
  with another URL. Found while planning, so `rote mcp import` resolves each
  one in the plan the user confirms (skip with a note, or `--rename`), never at
  the write. The rule for resolving them belongs to that PR.

Claude Code (this PR) reads user scope, local scope (nearest ancestor key of
`projects` in `~/.claude.json`), `.mcp.json` with its approval rules
(`enableAllProjectMcpServers`, `enabledMcpjsonServers`,
`disabledMcpjsonServers` across settings files and the `projects` entry;
committed `.claude/settings.json` approves only in a trusted folder; a disable
wins from anywhere), the `/mcp` toggle (`disabledMcpServers`), precedence
local > project > user with the loser marked shadowed, enabled plugins
(`installed_plugins.json` v2, `.mcp.json` wrapped or bare, `plugin.json`
`mcpServers` as a map, a JSON path, or a list; `.mcpb` bundles reported), and
`CLAUDE_CONFIG_DIR`. Shapes were checked against a real install by key name
only.

Managed policy (code.claude.com/docs/en/managed-mcp): `deniedMcpServers` is
merged from every settings file plus the system `managed-settings.json`
(`/Library/Application Support/ClaudeCode/` on macOS, `/etc/claude-code/` on
Linux, under `ClientContext.system_dir`), and a match blocks the candidate.
`serverName` matches exactly; `serverUrl` is read wider than Claude Code does
(scheme and port ignored, case-insensitive, one trailing `/` dropped on both
sides), because a deny that matches too much only skips an import and one
that matches too little imports a denied server. A `managed-mcp.json` marks every other server unused, since
Claude Code then loads only its servers. Not read, and said on every reading
(`SourceReading.limits`): MDM profiles, the Windows registry, claude.ai
server-managed settings, and `allowedMcpServers`; `managedMcpServers` is a
`problems` line when present.

### Every client (PR 2)

There is one module per client under `rote.clients`. Each docstring cites the
docs or source it was checked against on 2026-10-22. `rote.clients.common`
holds the parts they share:

- `Reader` parses JSON, JSONC (through json5), TOML and YAML. YAML uses 1.2
  booleans, so a Goose extension named `off` stays a string.
- `new_candidate`, `blocked` and `remote` apply the normalize rules in one
  place.
- `rank`: the whole entry wins by name, and the losers are marked shadowed.
- `merge_entries` serves clients that merge an entry field by field across
  layers.
- `switch_off` covers a disabled entry that still wins by name. It turns the
  server off instead of uncovering a lower entry.

| id | Files, highest precedence first | Remote shape | Placeholders | Off / approval |
| --- | --- | --- | --- | --- |
| `codex` | trusted `.codex/config.toml` (root to cwd), `$CODEX_HOME/config.toml`, `/etc/codex/config.toml`. Merged per field | `url` (Streamable HTTP only), `bearer_token_env_var`, `env_http_headers`, literal `http_headers` | none: env vars only by name | `enabled = false`. Servers in an untrusted project are listed inactive |
| `cursor` | `.cursor/mcp.json`, `~/.cursor/mcp.json` | `url`, `headers`, `auth.CLIENT_ID` | `${env:X}`. `${workspaceFolder}` and `${userHome}` unsupported | not in the file (a limit) |
| `vscode` | `.vscode/mcp.json` (`servers`), workspace `.mcp.json`, user `mcp.json` per OS (plus Insiders). Names compare case-insensitively | `type` http/sse, `url`, `headers`, `oauth.clientId` | `${env:X}`. `${input:id}` unsupported | `chat.mcp.access: "none"`. On/off state kept in VS Code storage is a limit |
| `copilot` | `.mcp.json` / `.github/mcp.json` from cwd up to the git root (closer wins), `$COPILOT_HOME/mcp-config.json` | `type` http/sse (default local), `oauthClientId`, `tools` | `$X`, `${X}`, `${X:-d}` | `disabledMcpServers` in `settings.json` |
| `gemini` | system, trusted `<cwd>/.gemini/settings.json`, `~/.gemini/settings.json`, system defaults, extensions | `httpUrl`, `url` + `type`, `tcp` | `$X`, `${X}`, `${X:-d}`. Extension variables unsupported | `mcp.allowed`, `mcp.excluded`, `admin.mcp.enabled`, `mcp-server-enablement.json`. `trustedFolders.json` |
| `opencode` | managed, `OPENCODE_CONFIG_CONTENT`, `OPENCODE_CONFIG_DIR`, `.opencode/`, project `opencode.json(c)`, `OPENCODE_CONFIG`, global. Deep-merged | `type: remote`, `url`, `headers`, `oauth` | `{env:X}`. `{file:...}` unsupported | `enabled: false` (v2: `disabled`) |
| `zed` | `.zed/settings.json`, user `settings.json` per OS | `url`, `headers`, `oauth.client_id` (Streamable HTTP) | none | `enabled: false` |
| `amp` | nearest `.amp/settings.json(c)`, user `settings.json(c)` | `url`, `transport`, `headers`, `oauth` | `${X}`. `${amp:...}` unsupported | workspace servers need `amp mcp approve`. Approvals are not verified (the format is undocumented), so they read unknown, and `allow: false` reads untrusted. An `amp.mcpPermissions` reject blocks |
| `goose` | user `config.yaml`, `GOOSE_ADDITIONAL_CONFIG_FILES`, `/etc/goose`. Merged per field | `type: streamable_http`, `uri`, `headers` (`sse` entries no longer load) | `${X}` / `$X`, filled only from the entry's `envs` / `env_keys` | `enabled: false` |
| `windsurf` | `.devin/mcp_config.local.json`, `.devin/mcp_config.json`, `~/.config/devin/mcp_config.json`, legacy `~/.codeium/windsurf*/` | `serverUrl` or `url`, `transport`, `headers` | `${env:X}`. `${file:...}` unsupported | `disabled: true` |
| `claude-desktop` | `claude_desktop_config.json` per OS | none: remote servers are claude.ai connectors | none | none |

**Trust** is part of the shared model: `ImportCandidate.trust` is
`trusted`, `untrusted` or `unknown`, with `trust_note` saying why.

- `trusted` means the scope has no gate, or rote read the gate and it passed.
- `untrusted` means rote read the gate and it failed. The entry is inactive
  too, so it wins no precedence. Examples: Codex `trust_level`, Gemini
  `trustedFolders.json`, an Amp entry with `"allow": false`.
- `unknown` means the client gates the scope on trust that rote cannot read,
  or the client loads it with no gate at all. The note reads "trust in
  <client> could not be checked: ...". This covers:
  - VS Code's Restricted Mode, Copilot CLI folder trust and Zed worktree
    trust.
  - Cursor and Devin project files, where the docs do not say.
  - OpenCode project config, which OpenCode loads without asking.
  - Amp workspace approvals. Amp does not document how an approval matches
    an entry, so an approval is never taken as proof.

An `unknown` entry stays listed as the client's. It is never `preselectable`,
and it appears in `needs_opt_in` beside a URL secret `risk`.
`rote mcp import` reads `needs_opt_in` and asks for each such server by name.

It fails closed by construction. A candidate in a repository-supplied scope
(`GATED_SCOPES`, today `project`) starts as `unknown`, with the note "no trust
gate was read for this scope". A source has two ways to change that:

- `trust_verified` after reading a gate that passed (Claude Code's project
  approval, Codex `trust_level`, Gemini trusted folders).
- `trust_unknown` or `trust_failed`, each with a note of its own.

`tests/test_clients_registry.py` gives every source a project fixture and
fails on any candidate still carrying the default note. A new source must
either get a fixture there or be listed as having no project scope.

`ClientContext.etc_dir` is the second machine-wide root, used by Codex,
Goose, Gemini CLI and OpenCode: `/etc` on every Unix, `%ProgramData%` on
Windows. Like `system_dir`, it is None under `ROTE_CLIENT_HOME_DIR`.

Every variable a source reads to find files is in `CLIENT_DIR_VARS`, and
isolation drops them all. `tests/test_clients_registry.py` derives that set
from the modules.

`ordered_sources(detected, ctx)` offers clients in this order: the one the
environment names, then the ones with config on this machine.

The interface cannot express the following. Each one becomes a `limits` line
or a `withheld` note:

- per-tool filters (`enabled_tools`, `includeTools`, `tools`)
- field-level merges across scopes, shown as one merged candidate at its
  highest file
- on/off state a client keeps outside its files (Cursor, VS Code)
- folder trust kept in app storage (VS Code, Zed, Copilot CLI): `trust: "unknown"`
- servers that live in an account: Claude Desktop connectors, Amp remote
  servers, Cursor team servers

## Skill discovery (`rote.skills`)

`rote.skills.discover(ctx, registry=)` lists every skill the user's
clients load, `<root>/<name>/SKILL.md`, one level down, and ranks them for
setup's skill step. Roots, in the order a duplicate is credited to:

| Client | Project (cwd up to the repo root) | User |
| --- | --- | --- |
| rote | the bundled examples that ship a skill, the demo `rote try` runs (`rote.demo.DEMO_NAME`, docs-digest) first | |
| Claude Code | `.claude/skills` | `<config dir>/skills` (`CLAUDE_CONFIG_DIR`), each enabled plugin's `skills/` |
| cross-client | `.agents/skills` | `~/.agents/skills` (Codex, Gemini CLI, OpenCode) |
| Codex | | `$CODEX_HOME/skills`, the older location Codex still reads |
| Gemini CLI | `.gemini/skills` | `~/.gemini/skills` (`GEMINI_CLI_HOME` moves `~`) |
| OpenCode | `.opencode/skills` | `$XDG_CONFIG_HOME/opencode/skills` |

Enabled plugins come from `rote.clients.claude_code.claude_code_layout`,
the same resolution the MCP source uses (`enabledPlugins` across user,
project and local settings, `installed_plugins.json` install paths). Not
read: Codex's admin `/etc/codex/skills`, Claude Code's managed and
`--add-dir` skills, and nested `skills/<a>/<b>/SKILL.md`, which no client
documents loading.

- **Dedupe** by resolved path: a skill symlinked into two roots is one
  skill whose `found_in` lists both.
- **Frontmatter**: the first 64 KB, `yaml.safe_load`, never raises. No
  frontmatter means the directory name (Claude Code's rule); bad YAML, an
  unclosed block or a non-mapping is a `problems` line. Displayed text
  drops control and format characters (bidi overrides, zero-width
  spaces) and is cut to 300 characters.
- **Hostile skills** (a `.claude/skills/<x>` in a cloned repository is
  scanned without anyone choosing it): each skill is read under a guard,
  so anything it raises (deeply nested YAML raised `RecursionError`) is
  that skill's `unreadable` entry, never the listing's crash. Its text is
  read through `rote.skill_files`, the same bounded reader compile's
  tool grounding and System-1 parser use: `SKILL.md` over 4 MB, or
  `references/` over 16 MB in total, is refused by `stat` before a byte is
  read (a 1.5 GB SKILL.md had cost 7.3 GB of RSS); a reference whose
  resolved path leaves the skill directory is skipped, and the walk never
  follows a symlinked directory (`references -> /` had walked the whole
  filesystem); `mcp-tools.json` over 16 MB is refused.
  `rote compile` runs the same checks first (`skill_files.preflight` in
  `Compiler._compile_inner`), so an oversized skill is a one-line
  `rote compile:` error before any driver runs, and a skipped reference
  that resolves outside the skill is one warning line.
- **Readiness**, from `tool_grounding.servers_named` against the
  registry and token store (`auth_status`): `bundled`, then `ready`
  (every named server registered with usable auth, or none named), then
  `needs_login` (fewest logins first), then `needs_server`, then
  `unreadable`. Each carries a reason sentence, the per-server facts, and
  the `next` commands (`rote mcp add`, `rote mcp login`).
- **Speed**: measured on a real home (178 skills: 56 Claude Code user, 61
  from enabled plugins, 39 `~/.agents`, 21 Codex, 1 bundled) at 0.33 s
  warm, 0.48 s for the first call in a process; every installed plugin's skills (199) take 0.88 s. The
  tool-table read used to run the whole System-1 parse; it is now
  `system1.parse.tool_registry`, a third of the cost.

`rote skills list [--json] [--limit N]` prints the ranking.

## Setup: steps, resumable from the world

```mermaid
sequenceDiagram
  participant U as user / agent
  participant S as rote setup
  participant St as Step (import, skill, proof)
  U->>S: rote setup [--json] [flags]
  S->>S: Mode.resolve, prompter_for(mode)
  loop each step in order
    S->>St: status(ctx)  (read-only facts)
    alt done
      S-->>U: "done already: ..."
    else pending
      S->>St: plan(ctx, prompter)
      S->>U: show plan
      S->>S: confirm_plan, apply_plan
      S->>S: stop on decline or failure
    end
  end
```

```python
class Step(Protocol):
    id: str; title: str; command: str   # the standalone command (rote mcp import, rote try)
    def status(self, ctx: SetupContext) -> StepStatus   # "done" | "pending", from facts
    def plan(self, ctx: SetupContext, prompter: Prompter) -> Plan   # no side effects

run_steps(steps, ctx, prompter, granted=()) -> list[StepReport]
```

`run_steps` writes `ctx.results[step.id] = {action_id: result}` after the
step's plan is applied, from each action's `run()` return value. A step never
writes results while planning: the plan may be declined or fail, and the next
step must read what happened, not what was intended.

Resumability comes from `status()` reading the world (servers in the registry,
token files, a chosen skill), not from a setup state file: an interrupted run
re-runs, the finished steps say "done already", and the rest continue. The one
fact the world does not hold, the chosen skill, is stored by the skill step in
user config when that PR lands. Each step is also its own idempotent command,
which builds the same `Step` and calls `run_steps([step], ...)`.

Under an agent, `rote setup --json` prints each step's status and its plan as
of now, with `command` for each action; the agent relays the choices and runs
the commands one step at a time (the `/rote:setup` plugin skill drives this).

`rote init` is the first command on this layer (`init_wizard.build_plan`
returns the Plan; `_cmd_init` confirms and applies). It becomes setup's
defaults step, and `rote init` stays as an alias.

## Deviations from the product design

- **`--agent-mode auto|yes|no`, not `--agent`.** `--agent` already names the
  compiler driver on `compile`, `analyze` and `init`, and setup absorbs init.
  `ROTE_AGENT_MODE` is its environment twin.
- **One missing answer at a time.** A non-interactive run stops at the first
  question without an answer, because later questions depend on earlier ones
  (`--runtime` is asked only for local hosting). `--yes` answers all of them.

## Open items the follow-up PRs own

- **`rote init --login` without a person** is refused (the device flow polls
  for a browser approval nobody can give). That is the rote cloud login;
  `rote mcp login` has the `--no-wait` / `--complete` split now (PR 4, see
  docs/mcp-client.md), and `rote login` needs its own, which can follow
  Stripe more closely (the device code, not a listener, is what the pending
  record would hold).
- **Two raw `input()` prompts remain in `cli.py`**: the baseline-input confirm
  ("Run the baseline with this input?") and the "Send them?" confirm. They
  predate this layer; move both onto `Prompter` + `Plan` (the baseline run is a
  paid `Launch` with its `Spend`).

- **`rote mcp login --no-wait`** (PR 4) lifts `rote mcp import --login`'s
  person-only rule, as it does `rote init --login`'s.

## `rote mcp import` (PR 3)

```
rote mcp import [--from <client>] [--only a,b | --servers-for <skill dir>]
                [--allow-url-secret <name>]... [--allow-untrusted <name>]...
                [--rename old=new]... [--replace <name>]... [--work-domain <d>]...
                [--login a,b | --no-login] [--yes] [--force] [--json]
```

`rote.setup.mcp_import.ImportPlanner` builds one `ImportPlan` (the `Plan` plus
a `Row` per server found, the collisions and the readings); `_cmd_mcp_import`
shows, confirms and applies it like `rote init`, and `McpImportStep` is the
same planner as a setup step (done when every preselectable, non-work server a
client uses is registered, by endpoint).

1. **Read** every detected `ClientSource` (detected client first), or the one
   `--from` names. A source that is not installed is skipped.
2. **Work domains, once.** `work_domains` is the first list setting in
   `rote.config` (`ConfigKey(kind="list")`, `resolve_list`): a YAML list of
   domains, strict like every key (a scalar, a scheme, a path, a port, a
   wildcard or an IP address is a `ConfigError`, and so is an empty
   `ROTE_WORK_DOMAINS`), shown by `rote config`. Unlike a string
   setting, every layer adds and none removes, so a committed project
   `rote.yaml` saying `work_domains: []` cannot switch off the user's own
   (Claude Code merges `deniedMcpServers` the same way). Asked when no layer
   holds the key and the answer matters: a person at the picker, or
   `--servers-for` (a default would pick). Only the user answers it: their
   own config holding the key, a `--work-domain` flag, or the prompt. A
   project `rote.yaml` or `ROTE_WORK_DOMAINS` adds domains to the run but is
   not an answer (a committed `work_domains: []` must not stand in for the
   user) and is never saved. The question has no default, so even `--yes`
   stops on it; `--work-domain none` answers "none". The answer (a flag or
   the prompt, nothing else) is saved to the user config as an Action in the
   same plan, by editing only the `work_domains` line (`set_in_place`), so
   comments and other lines stay; a layout the line edit cannot read back is
   rewritten whole as a marked replacement (`--force`). An explicit `--only`
   never asks.
3. **Pick.** One checkbox (flag `--only`): host, client and scope, `WORK`,
   inactive reason, "needs your yes: ..." and how many settings were not
   copied; a blocked server is a row, not a choice; an endpoint already
   registered is disabled. Nothing is preselected but what `--servers-for`
   names (`tool_grounding.servers_wanted`: `servers_named` plus what a compile
   would contact), and of those only `preselectable` ones off work domains.
   `--only` matches the client's name first, then a `client:scope:name` key,
   then the rote name; two different servers under one name is an error
   naming the keys.
4. **Per-server yes.** Every reason a candidate is not `preselectable`, other
   than being inactive in the client, needs its own yes by name
   (`opt_ins`): a URL secret (`--allow-url-secret`), unknown client trust
   (`--allow-untrusted`, "trust in <client> could not be checked", read from
   `ImportCandidate.trust` when a source sets it), and any reason this version
   does not know (`--allow-untrusted` too), so a new source's non-preselectable
   state is never imported on a default. A person gets a confirm whose default
   is no; nobody at the terminal gets `PromptRequired` naming
   `--allow-url-secret <name>` (a switch: `next` passes it bare), and `--yes`
   declines it.
5. **Plan.** At most three kinds of Action: the work-domain write, one
   registry write, and a login per chosen server.
   - The registry write is built from one `Snapshot` and merges into it:
     `add`, `present` (the endpoint is registered under any name), `changed`
     (same name and endpoint, other transport, headers or scopes: rote's entry
     is kept), `replace` (only with `--replace`), and the rows that write
     nothing (`not-importable`, `not-selected`, `declined`, `duplicate`). Only
     `add`/`replace` write. A replace loses content, so its `FileWrite` has
     `replaces=True` and needs `--force` without a person. File mode 0600.
     Re-running writes nothing new.
   - **Collision rule.** A picked server whose name is registered for another
     endpoint, or that wants the same name as another pick, is never written
     over: it is added under the first free `<name>_2`, `<name>_3`, ..., and
     the row says so with the two alternatives, `--rename old=new` (refused
     onto a taken name) and `--replace <name>`. URLs compare normalized
     (`normalize_server_url`), in `find_collisions` too.
   - Headers are written as templates (`Bearer ${ACME_TOKEN}`), never
     resolved; a note names each variable and whether this shell has it.
     Withheld settings (`oauth.clientId`, `headersHelper`, literal headers)
     are notes.
   - **Logins** are offered to a person only (a checkbox, flag `--login`,
     default: the picked servers that need OAuth and are not on a work
     domain), each its own Action: `Contact(host)` plus `Launch(your browser)`,
     command `rote mcp login <name>`, run after the registry write. A server
     with headers has no OAuth login (`rote mcp login` refuses one).
     `--login` without a person is an error until PR 4.

The human display prints the rows that change or cannot import and counts the
unpicked ones; `--json` lists every server (hosts, never URLs), and an
`action_required` stop carries what was found so an agent can relay the pick.

**`${NAME}` in registry headers** is filled at every connection by one rule,
`expand_headers` (Python helper, agent loop `--mcp-config`, `rote mcp
headers`, compile-time listing) and its TS twin `_HEADER_TEMPLATE_TS`, spliced
into the Node helper. Only the Node helper reads registry headers: the Workers
helper authenticates from provisioned OAuth secrets and the binding helper
through the platform. An unset or empty variable, or a value with a line break
or NUL, is a `RoteMcpConfigError` naming server, header and variable, never an
empty header and never a retry. Skill runs pass a templated header as a
`headersHelper`, so the value never lands in the `--mcp-config` file.
`tests/test_mcp_header_template_parity_e2e.py` runs the TS rule on the Python
table and both helpers against one real server.

## Remaining PRs

| # | PR | Builds on | Parallel with |
| --- | --- | --- | --- |
| 2 | Other client sources (Codex, Cursor, VS Code/Copilot, Gemini, OpenCode, Zed, Amp, Goose, Windsurf/Devin, Claude Desktop), one module each: **landed** | `ClientSource`, `normalize` | 3, 4, 5, 6 |
| 3 | `rote mcp import` (+ header expansion in the runtime helpers, work domains, collision rules): **landed** | `ClientSource`, `Plan`, `Prompter` | 2, 4, 5, 6 |
| 4 | `rote mcp login --no-wait` / `--complete` (Stripe's pending-file shape, atomic) | `Plan`, `report` | 2, 3, 5, 6 |
| 5 | Skill discovery and ranking (`rote.skills`, landed; see above) | `ClientContext`, `Prompter` | 2, 3, 4, 6 |
| 6 | Bundle docs-digest into examples/ and the wheel, with a recorded reference run | none | 2, 3, 4, 5 |
| 7 | `rote try` (precompiled proof, then the live path with `Spend(flag="--live")`, field-by-field report) | 6, `Plan` | 2, 3, 4, 5 |
| 8 | `rote setup` (steps from 3, 5, 7 plus init's defaults; first-run banner on bare `rote`) | 3, 4, 5, 7 | 9 once its JSON is fixed |
| 9 | `/rote:setup` plugin skill + AGENTS.md section | 8's `--json` shape | 10 |
| 10 | Fresh-install test: a real TTY (pty), inside Claude Code, and CI | 8 | 9 |
