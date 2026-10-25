"""``rote mcp import``: copy MCP servers from the user's AI clients into rote's registry.

The servers a person already uses live in their AI clients' configs
(Claude Code today; every :mod:`rote.clients` source the same way). This
reads them, shows what it found, lets the person pick, and plans the
writes, so the import is a plan the person confirms, never a side effect:

1. **Read** every detected client (``--from`` names one). Sources only
   read files (:mod:`rote.clients`).
2. **Ask** the user's work domains once, the first time it matters, and
   keep the answer in the user config (``work_domains``, a list setting).
3. **Pick**. Each server is shown with its host, client and scope, why it
   cannot be imported (blocked), why the client does not use it now
   (inactive), what was withheld, and a work flag. Nothing is preselected
   except the servers ``--servers-for <skill>`` says the skill needs, and
   never a work server or one whose URL path may carry a key
   (``ImportCandidate.risk``). A risky server needs its own yes, by name
   (``--allow-url-secret <name>``).
4. **Plan**. One write of the registry, additive and diff-aware: each
   picked server is added, skipped (already there), kept (changed in the
   client: ``--replace`` takes the client's), or renamed out of a
   collision. Then, for a person, a ``rote mcp login`` per server they
   choose, each its own action that contacts the server's host.

Re-running is additive: what is registered is never touched without
``--replace`` (plus ``--force``, since the plan then loses content).

Name collisions, the rule (decided in the plan, so the write never meets
one): a picked server whose name is already registered for another
endpoint, or that wants the same name as another picked server, is never
written over. It is added under the first free ``<name>_2``, ``<name>_3``
... and the plan says so; ``--rename old=new`` picks the name instead, and
``--replace <name>`` overwrites the registered entry. A rename onto a
taken name is refused. A server whose endpoint is already registered under
any name is "already registered", not a second entry.
"""

from __future__ import annotations

import io
import os
import re
import shlex
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from rote.clients import (
    ClientContext,
    ClientSource,
    Collision,
    ImportCandidate,
    SourceReading,
    find_collisions,
    ordered_sources,
    source,
)
from rote.clients.normalize import is_work_host
from rote.config import (
    ConfigError,
    ConfigLayers,
    ConfigValue,
    domain_problem,
    normalize_domain,
    parse_config_text,
    render_config,
    resolve_list,
    set_in_place,
)
from rote.interact.output import Output
from rote.interact.plan import Action, Contact, Launch, Plan, Snapshot, snapshot, write_file_action
from rote.interact.prompter import Choice, Prompter
from rote.mcp._runtime_helper import normalize_server_url
from rote.mcp.registry import (
    REGISTRY_FILE_MODE,
    McpRegistry,
    McpServerConfig,
    parse_registry,
    render_registry,
)
from rote.setup import SetupContext, StepStatus

#: The IR's ``mcp.server`` rule: a registry name is a binding's server name.
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

#: ``--work-domain none``: the answer "I have no work domains".
NO_WORK_DOMAINS = "none"

Status = Literal[
    "add",
    "replace",
    "present",
    "changed",
    "not-importable",
    "not-selected",
    "declined",
    "duplicate",
]


class McpImportError(ValueError):
    """Flags or files the import cannot plan from (exit 2, nothing written)."""


@dataclass(frozen=True)
class ImportFlags:
    """The answers given as flags. None: not given, so ask (or take the default)."""

    from_client: str | None = None
    #: ``--only a,b``: the pick, by name. None: not given.
    only: tuple[str, ...] | None = None
    #: ``--servers-for <skill dir>``: preselect what the skill needs.
    servers_for: Path | None = None
    allow_url_secret: tuple[str, ...] = ()
    #: ``--allow-untrusted``: servers whose client trust could not be checked,
    #: or that are not preselectable for any other reason a person must accept.
    allow_untrusted: tuple[str, ...] = ()
    #: ``--rename old=new`` pairs.
    rename: tuple[tuple[str, str], ...] = ()
    replace: tuple[str, ...] = ()
    #: ``--work-domain`` values (``none`` is the answer "no work domains").
    work_domains: tuple[str, ...] = ()
    #: ``--login a,b`` names; ``()`` for ``--no-login``; None: ask a person.
    login: tuple[str, ...] | None = None


@dataclass(frozen=True)
class Row:
    """What the plan does with one server found in a client."""

    candidate: ImportCandidate
    status: Status
    #: The rote name it is (or would be) registered under.
    name: str
    work: bool
    detail: str = ""

    @property
    def writes(self) -> bool:
        return self.status in ("add", "replace")

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.candidate.to_dict(),
            "status": self.status,
            "registered_as": self.name,
            "work": self.work,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class ImportPlan:
    plan: Plan
    rows: tuple[Row, ...]
    readings: tuple[SourceReading, ...]
    collisions: tuple[Collision, ...]
    work_domains: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "servers": [r.to_dict() for r in self.rows],
            "collisions": [c.to_dict() for c in self.collisions],
            "sources": [r.to_dict() for r in self.readings],
            "work_domains": list(self.work_domains),
        }


@dataclass(frozen=True)
class OptIn:
    """A reason a picked server needs its own yes, by name, before import."""

    reason: str
    #: The flag that gives that yes without a prompt (``--allow-url-secret``).
    flag: str


def opt_ins(c: ImportCandidate, client_label: str) -> tuple[OptIn, ...]:
    """Every per-server yes ``c`` needs: its ``needs_opt_in`` reasons, each
    with the flag that gives it (a URL secret, ``--allow-url-secret``; trust
    the client's gate did not let rote check, ``--allow-untrusted``). Being
    inactive in the client needs none: picking it is the person's own
    choice. Any other reason a server is not preselectable needs an opt-in
    too, so one a later version adds is never imported on a default or a
    bare ``--yes``."""
    if not c.importable or c.preselectable:
        return ()
    found: list[OptIn] = []
    if c.risk is not None:
        found.append(OptIn(c.risk, "--allow-url-secret"))
    if c.trust == "unknown":
        note = c.trust_note or f"trust in {client_label} could not be checked"
        found.append(OptIn(note, "--allow-untrusted"))
    if not found and c.inactive is None and c.trust != "untrusted":
        # Not preselectable, not inactive, no reason this version names.
        found.append(OptIn("rote could not check that this server is safe", "--allow-untrusted"))
    return tuple(found)


def candidate_key(c: ImportCandidate) -> str:
    """A server's identity among everything read: client, scope, plugin, name."""
    plugin = f"{c.plugin}:" if c.plugin else ""
    return f"{c.client}:{c.scope}:{plugin}{c.original_name}"


def _url_key(c: ImportCandidate) -> str | None:
    return None if c.url is None else normalize_server_url(c.url)


def _same_settings(c: ImportCandidate, config: McpServerConfig) -> bool:
    headers = {h.name: h.template for h in c.headers} or None
    scopes = list(c.scopes) or None
    return (config.transport, config.headers, config.scopes) == (c.transport, headers, scopes)


def _config_for(c: ImportCandidate) -> McpServerConfig:
    assert c.url is not None
    return McpServerConfig(
        url=c.url,
        transport=c.transport,
        headers={h.name: h.template for h in c.headers} or None,
        scopes=list(c.scopes) or None,
    )


def split_names(values: Sequence[str]) -> tuple[str, ...]:
    """Repeatable, comma-separated flag values as one tuple (``--only a,b --only c``)."""
    return tuple(n.strip() for v in values for n in v.split(",") if n.strip())


def _scope_label(c: ImportCandidate, label: Mapping[str, str]) -> str:
    where = label.get(c.client, c.client)
    if c.scope == "plugin":
        return f"{where} plugin {c.plugin}"
    return f"{where} {c.scope}"


@dataclass
class ImportPlanner:
    """Plans one import. Fields fill in as it goes, so a caller that stops
    on a missing answer can still show what was found (``found``)."""

    flags: ImportFlags
    prompter: Prompter
    out: Output
    ctx: ClientContext
    layers: ConfigLayers
    registry_path: Path
    user_config_path: Path
    #: The client the environment names (``rote.interact.agents``), offered first.
    detected_client: str | None = None
    #: Logs in to one server. Injected by tests; the CLI's is ``rote mcp login``'s.
    login: Callable[[str], object] | None = None
    #: Which skill servers ``--servers-for`` wants, from the candidate names.
    #: Injected by tests; defaults to ``tool_grounding.servers_wanted``.
    wanted_by_skill: Callable[[Path, Sequence[str]], Mapping[str, str]] | None = None

    readings: list[SourceReading] = field(default_factory=list)
    found: list[ImportCandidate] = field(default_factory=list)
    #: Picked servers whose opt-in the user did not give, with the one refused.
    declined: list[tuple[ImportCandidate, OptIn]] = field(default_factory=list)
    _labels: dict[str, str] = field(default_factory=dict)
    #: The servers offered (importable, one per endpoint), in reading order.
    primary: list[ImportCandidate] = field(default_factory=list)

    # ── reading ──

    def _sources(self) -> tuple[ClientSource, ...]:
        if self.flags.from_client is not None:
            try:
                return (source(self.flags.from_client),)
            except KeyError as e:
                raise McpImportError(str(e).strip("'\"")) from None
        return tuple(s for s in ordered_sources(self.detected_client) if s.detect(self.ctx))

    def _read(self) -> dict[str, str]:
        sources = self._sources()
        labels = {s.id: s.label for s in sources}
        for s in sources:
            reading = s.read(self.ctx)
            self.readings.append(reading)
            self.found.extend(reading.candidates)
            self.out.say(
                f"{s.label}: {len(reading.candidates)} server(s) in "
                f"{len(reading.files_read)} file(s)"
            )
            for problem in reading.problems:
                self.out.warn(f"{s.label}: {problem}")
        if not sources:
            self.out.say("No AI client config found on this machine (rote reads: Claude Code).")
        return labels

    # ── work domains ──

    def _work_domains(self) -> tuple[tuple[str, ...], Action | None]:
        given = tuple(normalize_domain(d) for d in self.flags.work_domains)
        if NO_WORK_DOMAINS in given and len(given) > 1:
            raise McpImportError("--work-domain none cannot be combined with domains")
        flag_domains = tuple(d for d in given if d != NO_WORK_DOMAINS)
        try:
            resolved = resolve_list("work_domains", flag_domains, layers=self.layers)
        except ConfigError as e:
            raise McpImportError(str(e)) from None
        # Answered only by the user: their own config, a flag, or the prompt.
        # A project rote.yaml or ROTE_WORK_DOMAINS adds domains, but `[]`
        # there is not the user saying they have none.
        answered = "work_domains" in self.layers.user or bool(given)
        needed = (self.prompter.interactive or self.flags.servers_for is not None) and any(
            c.importable for c in self.found
        )
        new: tuple[str, ...] = flag_domains
        asked = False
        if not answered and needed:
            raw = self.prompter.text(
                "Your work domains, comma-separated (their servers are never preselected; "
                "blank for none)",
                flag="--work-domain",
                default=None,
                validate=_domains_problem,
            )
            new = _parse_domains(raw)
            asked = True
        domains = tuple(dict.fromkeys((*resolved.values, *new)))
        user_has = self.layers.user.get("work_domains")
        # Saved only from what the user said here: a flag or the answer.
        if not (given or asked) or (isinstance(user_has, tuple) and set(new) <= set(user_has)):
            return domains, None
        return domains, self._config_action(new)

    def _config_action(self, new: tuple[str, ...]) -> Action:
        read = snapshot(self.user_config_path)
        text = read.text()
        try:
            values: dict[str, ConfigValue] = (
                parse_config_text(text, self.user_config_path) if text is not None else {}
            )
        except ConfigError as e:
            raise McpImportError(str(e)) from None
        kept = values.get("work_domains", ())
        assert isinstance(kept, tuple)
        merged = tuple(dict.fromkeys((*kept, *new)))
        shown = ", ".join(merged) if merged else "none"
        # Only the work_domains line changes; the user's comments and other
        # lines stay. A layout the line edit cannot handle is rewritten
        # whole, and that loses comments, so it is a marked replacement.
        content = (
            set_in_place(text, "work_domains", merged, self.user_config_path)
            if text is not None
            else render_config({"work_domains": merged}, origin=str(self.user_config_path))
        )
        replaces = content is None
        if content is None:
            values["work_domains"] = merged
            content = render_config(values, origin=str(self.user_config_path))
        return write_file_action(
            id="write-work-domains",
            title="Save your work domains",
            read=read,
            content=content,
            summary=f"work_domains: {shown}"
            + (" (rewrites the file: its comments are lost)" if replaces else ""),
            replaces=replaces,
            command="rote mcp import "
            + " ".join(f"--work-domain {d}" for d in (merged or (NO_WORK_DOMAINS,))),
        )

    # ── picking ──

    def _wanted(self, candidates: Sequence[ImportCandidate]) -> Mapping[str, str]:
        if self.flags.servers_for is None:
            return {}
        skill = self.flags.servers_for
        if not (skill / "SKILL.md").is_file():
            raise McpImportError(f"--servers-for {skill}: no SKILL.md there")
        names = sorted({n for c in candidates for n in (c.name, c.original_name)})
        if self.wanted_by_skill is not None:
            return self.wanted_by_skill(skill, names)
        from rote.compiler.tool_grounding import servers_wanted

        return servers_wanted(skill, names)

    def build(self) -> ImportPlan:
        labels = self._read()
        reg_read = snapshot(self.registry_path)
        try:
            registry = parse_registry(reg_read.data)
        except ValueError as e:
            raise McpImportError(f"{self.registry_path}: not a rote MCP registry: {e}") from None
        domains, config_action = self._work_domains()
        registered_by_url = {
            normalize_server_url(cfg.url): name for name, cfg in sorted(registry.servers.items())
        }

        rows: dict[str, Row] = {}
        primary: list[ImportCandidate] = []
        seen_urls: dict[tuple[str, str], str] = {}
        for c in self.found:
            key = candidate_key(c)
            work = is_work_host(c.host, domains)
            if not c.importable:
                rows[key] = Row(c, "not-importable", c.name, work, c.blocked or "")
                continue
            url = _url_key(c)
            assert url is not None
            ident = (url, repr(sorted((h.name, h.template) for h in c.headers)))
            if ident in seen_urls:
                rows[key] = Row(c, "duplicate", c.name, work, f"same server as {seen_urls[ident]}")
                continue
            seen_urls[ident] = f"{_scope_label(c, labels)} {c.original_name}"
            primary.append(c)
        self.primary = primary

        wanted = self._wanted(primary)
        picked = self._pick(primary, registry, registered_by_url, domains, wanted, labels)
        self._labels = labels
        picked = self._opt_in(picked)
        for c in primary:
            key = candidate_key(c)
            if key in rows:
                continue
            if c not in picked and c not in [d for d, _ in self.declined]:
                already = registered_by_url.get(_url_key(c) or "")
                work = is_work_host(c.host, domains)
                if already is not None:
                    detail = (
                        "already registered" if already == c.name else f"registered as {already}"
                    )
                    rows[key] = Row(c, "present", already, work, detail)
                else:
                    rows[key] = Row(c, "not-selected", c.name, work)

        resolved, collisions = self._resolve_names(picked, registry, registered_by_url, domains)
        for row in resolved:
            rows[candidate_key(row.candidate)] = row
        ordered = tuple(rows[candidate_key(c)] for c in self.found)

        actions: list[Action] = []
        if config_action is not None:
            actions.append(config_action)
        writes = [r for r in ordered if r.writes]
        if writes:
            actions.append(self._registry_action(reg_read, registry, writes))
        actions.extend(self._login_actions(writes))
        picked_keys = {candidate_key(c) for c in (*picked, *(d for d, _ in self.declined))}
        notes = self._notes(
            [r for r in ordered if candidate_key(r.candidate) in picked_keys], writes, actions
        )
        plan = Plan(title="Import MCP servers into rote", actions=tuple(actions), notes=notes)
        return ImportPlan(plan, ordered, tuple(self.readings), collisions, domains)

    def _selector(self, c: ImportCandidate, primary: Sequence[ImportCandidate]) -> str:
        """What ``--only`` takes for ``c``: its name, or its key when two share it."""
        same = [p for p in primary if p.original_name == c.original_name]
        return c.original_name if len(same) == 1 else candidate_key(c)

    def _pick(
        self,
        primary: Sequence[ImportCandidate],
        registry: McpRegistry,
        registered_by_url: Mapping[str, str],
        domains: tuple[str, ...],
        wanted: Mapping[str, str],
        labels: Mapping[str, str],
    ) -> list[ImportCandidate]:
        if self.flags.only is not None:
            return self._pick_by_name(primary)
        choices: list[Choice[str]] = []
        default: list[str] = []
        by_selector: dict[str, ImportCandidate] = {}
        for c in primary:
            selector = self._selector(c, primary)
            by_selector[selector] = c
            work = is_work_host(c.host, domains)
            already = registered_by_url.get(_url_key(c) or "")
            disabled = None
            hints = [c.host or "", _scope_label(c, labels)]
            if already is not None and already != c.name:
                disabled = f"already in rote's registry as {already}"
            elif already is not None and _same_settings(c, registry.servers[already]):
                disabled = "already in rote's registry"
            elif already is not None:
                hints.append(f"registered with other settings (--replace {c.name} to take these)")
            if work:
                hints.append("WORK")
            if c.inactive:
                hints.append(f"inactive: {c.inactive}")
            for need in opt_ins(c, labels.get(c.client, c.client)):
                hints.append(f"needs your yes: {need.reason}")
            if c.withheld:
                hints.append(f"{len(c.withheld)} setting(s) not copied")
            why = wanted.get(c.name) or wanted.get(c.original_name)
            if why:
                hints.append(f"the skill needs it ({why})")
            choices.append(Choice(selector, c.name, "  ·  ".join(h for h in hints if h), disabled))
            if why and disabled is None and c.preselectable and not work:
                default.append(selector)
        if not any(choice.disabled is None for choice in choices):
            return []
        answer = self.prompter.checkbox(
            "Which servers should rote import?", choices, flag="--only", default=default
        )
        return [by_selector[s] for s in answer]

    def _pick_by_name(self, primary: Sequence[ImportCandidate]) -> list[ImportCandidate]:
        assert self.flags.only is not None
        picked: list[ImportCandidate] = []
        for wanted in self.flags.only:
            # The client's own name first, then a key, then the rote name:
            # `acme_crm` means the server called that, not `acme-crm`, whose
            # rote name it also is.
            matches = (
                [c for c in primary if c.original_name == wanted]
                or [c for c in primary if candidate_key(c) == wanted]
                or [c for c in primary if c.name == wanted]
            )
            blocked = [
                c for c in self.found if not c.importable and wanted in (c.name, c.original_name)
            ]
            if not matches and blocked:
                raise McpImportError(f"--only {wanted}: not importable: {blocked[0].blocked}")
            if not matches:
                known = ", ".join(sorted({self._selector(c, primary) for c in primary})) or "none"
                raise McpImportError(f"--only {wanted}: no such server was found; found: {known}")
            active = [c for c in matches if c.inactive is None] or matches
            if len({_url_key(c) for c in active}) > 1:
                keys = ", ".join(candidate_key(c) for c in active)
                raise McpImportError(
                    f"--only {wanted} names {len(active)} different servers ({keys}); "
                    "pass one of those keys, or --from <client>"
                )
            if active[0] not in picked:
                picked.append(active[0])
        return picked

    def _opt_in(self, picked: list[ImportCandidate]) -> list[ImportCandidate]:
        """Each picked server's per-server yeses (:func:`opt_ins`): its flag,
        else a question whose default is no, so a bare ``--yes`` declines."""
        allowed = {
            "--allow-url-secret": set(self.flags.allow_url_secret),
            "--allow-untrusted": set(self.flags.allow_untrusted),
        }
        known = {n for c in self.found for n in (c.name, c.original_name)}
        for flag, names in allowed.items():
            unknown = sorted(names - known)
            if unknown:
                raise McpImportError(f"{flag} {unknown[0]}: no such server was found")
        kept: list[ImportCandidate] = []
        for c in picked:
            label = self._labels.get(c.client, c.client)
            refused = None
            for need in opt_ins(c, label):
                if {c.name, c.original_name} & allowed[need.flag]:
                    continue
                if not self.prompter.confirm(
                    f"{c.name}: {need.reason}. Import it anyway?",
                    flag=f"{need.flag} {c.name}",
                    default=False,
                ):
                    refused = need
                    break
            if refused is None:
                kept.append(c)
            else:
                self.declined.append((c, refused))
        return kept

    # ── names and collisions ──

    def _resolve_names(
        self,
        picked: Sequence[ImportCandidate],
        registry: McpRegistry,
        registered_by_url: Mapping[str, str],
        domains: tuple[str, ...],
    ) -> tuple[list[Row], tuple[Collision, ...]]:
        renames: dict[str, str] = {}
        for old, new in self.flags.rename:
            if not _IDENTIFIER.fullmatch(new):
                raise McpImportError(f"--rename {old}={new}: {new!r} is not an identifier")
            renames[old] = new
        replacing = set(self.flags.replace)
        renamed = [
            replace(c, name=renames.get(c.original_name, renames.get(c.name, c.name)))
            for c in picked
        ]
        used_renames = {k for k in renames for c in picked if k in (c.original_name, c.name)}
        if set(renames) - used_renames:
            raise McpImportError(
                f"--rename {sorted(set(renames) - used_renames)[0]}: not a server being imported"
            )
        registered_urls = {name: cfg.url for name, cfg in registry.servers.items()}
        collisions = find_collisions(renamed, registered_urls)

        rows: list[Row] = []
        taken: set[str] = set(registry.servers)
        replaced: set[str] = set()
        for original, c in zip(picked, renamed, strict=True):
            work = is_work_host(c.host, domains)
            explicit = c.name != original.name
            url = _url_key(c) or ""
            existing_name = registered_by_url.get(url)
            if existing_name is not None:
                config = registry.servers[existing_name]
                if existing_name != c.name:
                    rows.append(
                        Row(
                            original,
                            "present",
                            existing_name,
                            work,
                            f"this endpoint is registered as {existing_name}",
                        )
                    )
                elif _same_settings(c, config):
                    rows.append(Row(original, "present", c.name, work, "already registered"))
                elif c.name in replacing:
                    replaced.add(c.name)
                    rows.append(
                        Row(
                            original,
                            "replace",
                            c.name,
                            work,
                            "replaces the registered entry's settings with the client's",
                        )
                    )
                else:
                    rows.append(
                        Row(
                            original,
                            "changed",
                            c.name,
                            work,
                            "registered with other settings; kept rote's entry "
                            f"(--replace {c.name} takes the client's)",
                        )
                    )
                taken.add(c.name)
                continue
            if c.name in registry.servers and c.name not in replaced:
                other = urlsplit(registry.servers[c.name].url).hostname or "another endpoint"
                if c.name in replacing:
                    replaced.add(c.name)
                    rows.append(
                        Row(
                            original,
                            "replace",
                            c.name,
                            work,
                            f"replaces the entry registered for {other}",
                        )
                    )
                    continue
                if explicit:
                    raise McpImportError(
                        f"--rename: {c.name} is already registered for {other}; pick a free name"
                    )
                name = _free_name(c.name, taken)
                taken.add(name)
                rows.append(
                    Row(
                        original,
                        "add",
                        name,
                        work,
                        f"{c.name} is registered for {other}; added as {name} "
                        f"(--rename {original.original_name}=<name> or --replace {c.name} instead)",
                    )
                )
                continue
            if c.name in taken:
                if explicit:
                    raise McpImportError(
                        f"--rename: {c.name} is already taken by another server in this import"
                    )
                name = _free_name(c.name, taken)
                taken.add(name)
                rows.append(
                    Row(
                        original,
                        "add",
                        name,
                        work,
                        f"another picked server is also named {c.name}; added as {name} "
                        f"(--rename {original.original_name}=<name> instead)",
                    )
                )
                continue
            taken.add(c.name)
            if explicit:
                detail = f"renamed from {original.name}"
            elif original.renamed:
                detail = f"{original.original_name} in the client; rote names are identifiers"
            else:
                detail = ""
            rows.append(Row(original, "add", c.name, work, detail))
        for c, need in self.declined:
            rows.append(
                Row(
                    c,
                    "declined",
                    c.name,
                    is_work_host(c.host, domains),
                    f"{need.reason}; pass {need.flag} {c.name} to import it",
                )
            )
        unused = replacing - {r.name for r in rows if r.status == "replace"}
        if unused:
            raise McpImportError(
                f"--replace {sorted(unused)[0]}: not a registered server this import would change"
            )
        return rows, collisions

    # ── actions ──

    def _registry_action(
        self, read: Snapshot, registry: McpRegistry, writes: Sequence[Row]
    ) -> Action:
        merged = registry.model_copy(deep=True)
        for row in writes:
            merged.servers[row.name] = _config_for(row.candidate)
        added = [r.name for r in writes if r.status == "add"]
        replaced = [r.name for r in writes if r.status == "replace"]
        parts = []
        if added:
            parts.append(f"add {', '.join(added)}")
        if replaced:
            parts.append(f"replace {', '.join(replaced)}")
        return write_file_action(
            id="write-registry",
            title=f"Register {len(writes)} MCP server{'s' if len(writes) != 1 else ''}",
            read=read,
            content=render_registry(merged),
            summary="; ".join(parts),
            replaces=bool(replaced),
            file_mode=REGISTRY_FILE_MODE,
            command=self._command(writes),
        )

    def _command(self, writes: Sequence[Row]) -> str:
        words = ["rote", "mcp", "import"]
        if self.flags.from_client:
            words += ["--from", self.flags.from_client]
        words += ["--only", ",".join(self._selector(r.candidate, self.primary) for r in writes)]
        for r in writes:
            if r.name != r.candidate.name:
                words += ["--rename", f"{r.candidate.original_name}={r.name}"]
            if r.status == "replace":
                words += ["--replace", r.name]
            for need in opt_ins(r.candidate, self._labels.get(r.candidate.client, "")):
                words += [need.flag, r.candidate.name]
        words += ["--no-login", "--yes"]
        if any(r.status == "replace" for r in writes):
            words.append("--force")
        return " ".join(shlex.quote(w) for w in words)

    def _login_actions(self, writes: Sequence[Row]) -> list[Action]:
        loginable = [r for r in writes if not r.candidate.headers]
        names = {r.name for r in loginable}
        if self.flags.login is not None:
            chosen = list(self.flags.login)
            unknown = [n for n in chosen if n not in names]
            if unknown:
                raise McpImportError(
                    f"--login {unknown[0]}: not a server this import registers for OAuth "
                    f"(registers: {', '.join(sorted(names)) or 'none'})"
                )
            if chosen and not self.prompter.interactive:
                # The browser login waits for a person to approve it; with nobody
                # at the terminal it would block. Lifted when `rote mcp login`
                # gets --no-wait / --complete (docs/design/first-run.md, PR 4).
                raise McpImportError(
                    "--login needs a person at the terminal (the login waits for approval in "
                    "a browser); run rote mcp login <name> in a terminal afterwards"
                )
        elif self.prompter.interactive and loginable:
            choices = [
                Choice(
                    r.name,
                    r.name,
                    (r.candidate.host or "") + ("  ·  WORK" if r.work else ""),
                )
                for r in loginable
            ]
            chosen = self.prompter.checkbox(
                "Log in now? Each one opens your browser (or later: rote mcp login <name>)",
                choices,
                flag="--login",
                default=[r.name for r in loginable if not r.work],
            )
        else:
            chosen = []
        by_name = {r.name: r for r in loginable}
        return [self._login_action(by_name[n]) for n in chosen]

    def _login_action(self, row: Row) -> Action:
        host = row.candidate.host
        assert host is not None
        return login_action(row.name, host, self.login)

    def _notes(
        self, rows: Sequence[Row], writes: Sequence[Row], actions: Sequence[Action]
    ) -> tuple[str, ...]:
        notes: list[str] = []
        logging_in = {a.id.removeprefix("login-") for a in actions if a.id.startswith("login-")}
        for r in rows:
            if r.status in ("present", "changed", "declined", "duplicate"):
                notes.append(f"{r.candidate.original_name}: {r.status}: {r.detail}")
            elif r.writes and r.detail:
                notes.append(f"{r.candidate.original_name}: {r.detail}")
        for r in writes:
            for item in r.candidate.withheld:
                notes.append(f"{r.name}: not copied: {item}")
            for header in r.candidate.headers:
                missing = [v for v in header.env_vars if not self._env_has(v)]
                notes.append(
                    f"{r.name}: header {header.name} reads {', '.join(header.env_vars)} from "
                    "the environment each run starts in"
                    + (f" ({', '.join(missing)} not set in this shell)" if missing else "")
                )
            if not r.candidate.headers and r.name not in logging_in:
                notes.append(f"{r.name}: next: rote mcp login {r.name}")
        return tuple(notes)

    @staticmethod
    def _env_has(var: str) -> bool:
        return bool(os.environ.get(var))


def _free_name(name: str, taken: set[str]) -> str:
    n = 2
    while f"{name}_{n}" in taken:
        n += 1
    return f"{name}_{n}"


def _parse_domains(raw: str) -> tuple[str, ...]:
    items = [normalize_domain(d) for d in raw.split(",") if d.strip()]
    if items == [NO_WORK_DOMAINS]:
        return ()
    return tuple(dict.fromkeys(items))


def _domains_problem(raw: str) -> str | None:
    for item in _parse_domains(raw):
        problem = domain_problem(item)
        if problem is not None:
            return f"{item}: {problem}"
    return None


def _login_server(name: str) -> None:
    from rote.mcp.parked import login_server

    login_server(name)


def login_action(name: str, host: str, login: Callable[[str], object] | None = None) -> Action:
    """The blocking browser login for one registered server, as an Action:
    ``rote mcp import``'s and ``rote setup``'s login step's, for a person.
    ``login`` is injected by tests; the default is ``rote mcp login``'s."""
    run_login = login if login is not None else _login_server

    def run() -> str:
        run_login(name)
        return name

    return Action(
        id=f"login-{name}",
        title=f"Log in to {name}",
        effects=(
            Contact(host=host.split(":", 1)[0], purpose="OAuth discovery and sign-in"),
            Launch(program="your browser", purpose=f"approve rote's access to {name}"),
        ),
        run=run,
        command=f"rote mcp login {name}",
    )


def render_rows(rows: Sequence[Row]) -> list[str]:
    """The diff against rote's registry for a person: one line per server
    the plan adds, replaces, skips or keeps, and per server it cannot
    import; the servers left unpicked are counted, not listed (``--json``
    lists every one)."""
    marks = {
        "add": "+",
        "replace": "!",
        "present": "=",
        "changed": "~",
        "not-importable": "x",
        "declined": "?",
        "duplicate": "=",
    }
    shown = [r for r in rows if r.status in marks]
    unpicked = [r for r in rows if r.status == "not-selected"]
    if not shown and not unpicked:
        return []
    lines = [
        "Servers (+ add, ! replace, = already there, ~ changed, ? declined, x not importable):"
    ]
    width = max((len(r.name) for r in shown), default=0)
    for r in shown:
        host = r.candidate.host or "-"
        work = "  WORK" if r.work else ""
        lines.append(
            f"  {marks[r.status]} {r.name.ljust(width)}  {host}{work}  {r.status}"
            + (f": {r.detail}" if r.detail else "")
        )
    if unpicked:
        at_work = sum(1 for r in unpicked if r.work)
        lines.append(
            f"  {len(unpicked)} more not picked"
            + (f" ({at_work} on your work domains)" if at_work else "")
        )
    return lines


class McpImportStep:
    """``rote mcp import`` as a ``rote setup`` step (:mod:`rote.setup`)."""

    id = "mcp-import"
    title = "Import MCP servers from your AI clients"
    command = "rote mcp import"

    def __init__(self, make_planner: Callable[[Prompter, Output], ImportPlanner]) -> None:
        self._make_planner = make_planner

    def status(self, ctx: SetupContext) -> StepStatus:
        """Done when every server a client uses, with nothing to opt into and
        not on a work domain, is registered (by endpoint)."""
        from rote.interact.prompter import NonInteractivePrompter

        # Read quietly: plan() reads the clients again and says what it found.
        quiet = Output(ctx.mode, stdout=io.StringIO(), stderr=io.StringIO())
        planner = self._make_planner(NonInteractivePrompter(why="reading status"), quiet)
        planner._read()
        try:
            registry = parse_registry(snapshot(planner.registry_path).data)
        except ValueError:
            return StepStatus("pending", "rote's MCP registry does not parse")
        registered = {normalize_server_url(c.url) for c in registry.servers.values()}
        domains = resolve_list("work_domains", layers=planner.layers).values
        missing = [
            c
            for c in planner.found
            if c.preselectable
            and not is_work_host(c.host, domains)
            and _url_key(c) not in registered
        ]
        if missing:
            return StepStatus("pending", f"{len(missing)} server(s) not in rote's registry yet")
        return StepStatus("done", f"{len(registry.servers)} server(s) registered")

    def plan(self, ctx: SetupContext, prompter: Prompter) -> Plan:
        planned = self._make_planner(prompter, ctx.out).build()
        for line in render_rows(planned.rows):
            ctx.out.say(line)
        return planned.plan
