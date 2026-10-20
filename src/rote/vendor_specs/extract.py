"""The vendor requests a generated module makes, read from its AST (no code runs).

Two call shapes, the two the api backend's compiles actually write:

* **Raw HTTP** through ``httpx`` / ``requests``: ``httpx.get(url, params=...)``
  (or ``get`` imported from either), ``client.get(...)`` on an
  ``httpx.Client`` / ``AsyncClient``, a ``requests.Session`` or google-auth's
  ``AuthorizedSession``, however the class was imported, and
  ``.request("GET", url, ...)``. The URL is rebuilt as a template from string
  literals, f-strings, ``.format(...)``, ``+`` and module- or function-level
  string constants; anything computed becomes ``{?}``.
* **Google's client library**: ``build("gmail", "v1", ...)`` then
  ``service.users().threads().list(userId=..., q=...)``, which names a
  Discovery method id (``gmail.users.threads.list``) directly.

For each request it also collects the fields the code reads off the
response (``body.get("threads")``, then ``t["snippet"]`` for each ``t`` in
it), so they can be checked against the response schema.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Generic, Literal, TypeVar

_HTTP_VERBS = {"get": "GET", "post": "POST", "put": "PUT", "patch": "PATCH", "delete": "DELETE"}
#: Module-level request functions, by their full dotted name.
_HTTP_FUNCTIONS = {f"{m}.{v}" for m in ("httpx", "requests") for v in (*_HTTP_VERBS, "request")}
#: Classes whose instances send requests with ``.get(...)`` / ``.request(...)``.
_CLIENT_FACTORIES = {
    "httpx.Client",
    "httpx.AsyncClient",
    "requests.Session",
    "requests.sessions.Session",
    "google.auth.transport.requests.AuthorizedSession",
}

#: An argument value the AST cannot pin down.
UNKNOWN = object()


@dataclass(frozen=True)
class Expr:
    """A value known only as a template: what it is built from, not what it is.

    ``{name}`` is a name in the function (a parameter the node's inputs bind),
    ``{$env:VAR}`` an environment variable the code reads, ``{?}`` anything
    else. ``"Bearer {$env:GMAIL_TOKEN}"`` is an Authorization header read
    from the environment. The live check fills a template from the node's
    inputs and the user's environment; the spec check treats it as unknown.
    """

    template: str


_T = TypeVar("_T")


@dataclass
class CallSite:
    """One vendor request in a module."""

    line: int
    function: str | None
    kind: Literal["http", "google_client"]
    #: HTTP verb (``http``) or ``None`` (``google_client``: the method id decides).
    method: str | None = None
    #: URL template (``http``), ``{name}`` for formatted pieces, ``{?}`` for
    #: unknown ones; ``None`` when nothing static is left.
    url: str | None = None
    #: ``(api, version)`` from ``build(...)`` and the resource chain, for
    #: ``google_client``: the Discovery method id is
    #: ``<api>.<chain...>.<method>``.
    google_api: tuple[str, str] | None = None
    google_chain: tuple[str, ...] = ()
    #: Parameter name -> literal value, or :data:`UNKNOWN`.
    params: dict[str, object] = field(default_factory=dict)
    #: Whether every parameter the call sends is known (no ``**kwargs``).
    params_complete: bool = True
    body_keys: tuple[str, ...] = ()
    #: Response field paths the code reads: ``("threads", "[]", "id")``.
    response_reads: list[tuple[tuple[str, ...], int]] = field(default_factory=list)
    #: Request headers (``http``): name -> literal or :class:`Expr`; ``None``
    #: when the call passes headers the AST cannot read.
    headers: dict[str, object] | None = field(default_factory=dict)
    #: The call passes an ``auth=`` object (requests' HTTPBasicAuth...).
    auth_kwarg: bool = False

    @property
    def google_method_id(self) -> str | None:
        if self.google_api is None or not self.google_chain:
            return None
        return ".".join((self.google_api[0], *self.google_chain))


def extract(source: str) -> list[CallSite]:
    """Every vendor request ``source`` makes, in line order."""
    tree = ast.parse(source)
    module_consts = _string_consts(tree.body)
    sites: list[CallSite] = []
    module_names = _module_aliases(tree)
    factories = _build_factories(tree, module_names)
    dict_helpers = _dict_helpers(tree, module_consts)
    for scope in [tree, *[n for n in ast.walk(tree) if isinstance(n, _FUNCS)]]:
        consts = module_consts if scope is tree else _string_consts(scope.body, module_consts)
        name = scope.name if isinstance(scope, _FUNCS) else None
        sites += _Scope(scope, name, consts, module_names, factories, dict_helpers).sites()
    sites.sort(key=lambda s: s.line)
    return sites


_FUNCS = (ast.FunctionDef, ast.AsyncFunctionDef)


def _module_aliases(tree: ast.Module) -> dict[str, str]:
    """Local name -> module, for ``import httpx`` / ``import requests as r``
    and ``from googleapiclient.discovery import build``."""
    out: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                out[a.asname or a.name] = a.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            for a in node.names:
                out[a.asname or a.name] = f"{node.module}.{a.name}"
    return out


def _dotted(node: ast.expr, modules: dict[str, str]) -> str | None:
    """The full dotted name ``node`` refers to, through the module's imports.

    ``Client`` after ``from httpx import Client`` is ``httpx.Client``;
    ``gr.AuthorizedSession`` after ``import google.auth.transport.requests
    as gr`` is ``google.auth.transport.requests.AuthorizedSession``.
    """
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(modules.get(node.id, node.id))
    return ".".join(reversed(parts))


def _is_build(func: ast.expr, modules: dict[str, str]) -> bool:
    name = (
        modules.get(func.id, func.id)
        if isinstance(func, ast.Name)
        else func.attr
        if isinstance(func, ast.Attribute)
        else None
    )
    return name in ("googleapiclient.discovery.build", "build")


def _build_factories(tree: ast.Module, modules: dict[str, str]) -> dict[str, tuple[int, int]]:
    """Helpers that wrap ``build``: ``def _service(api, version, ...): return build(api, version)``.

    Name -> the positions of the ``api`` and ``version`` parameters, so a
    call ``_service("gmail", "v1", ...)`` names a service like ``build`` does.
    """
    out: dict[str, tuple[int, int]] = {}
    for fn in ast.walk(tree):
        if not isinstance(fn, _FUNCS):
            continue
        params = [a.arg for a in fn.args.args]
        for node in ast.walk(fn):
            if not (
                isinstance(node, ast.Return)
                and isinstance(node.value, ast.Call)
                and _is_build(node.value.func, modules)
                and len(node.value.args) >= 2
            ):
                continue
            api, version = node.value.args[0], node.value.args[1]
            if (
                isinstance(api, ast.Name)
                and isinstance(version, ast.Name)
                and api.id in params
                and version.id in params
            ):
                out[fn.name] = (params.index(api.id), params.index(version.id))
    return out


def _env_var(node: ast.expr) -> str | None:
    """``VAR`` for ``os.environ["VAR"]``, ``os.environ.get("VAR")`` or ``os.getenv("VAR")``."""
    if (
        isinstance(node, ast.Subscript)
        and _attr_chain(node.value) == ("os", "environ")
        and isinstance(node.slice, ast.Constant)
        and isinstance(node.slice.value, str)
    ):
        return node.slice.value
    if (
        isinstance(node, ast.Call)
        and _attr_chain(node.func) in (("os", "environ", "get"), ("os", "getenv"))
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    ):
        return node.args[0].value
    return None


def _attr_chain(node: ast.expr) -> tuple[str, ...]:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return tuple(reversed(parts))


def _dict_helpers(
    tree: ast.Module, consts: dict[str, str]
) -> dict[str, tuple[ast.Dict, dict[str, str]]]:
    """Module functions that build a dict and return it: ``def _headers(): return {...}``.

    ``headers=_headers()`` then names that dict; its values are templates in
    the helper's own scope (its environment reads included).
    """
    out: dict[str, tuple[ast.Dict, dict[str, str]]] = {}
    for fn in tree.body:
        if not isinstance(fn, _FUNCS) or not fn.body:
            continue
        last = fn.body[-1]
        if isinstance(last, ast.Return) and isinstance(last.value, ast.Dict):
            out[fn.name] = (last.value, _string_consts(fn.body, consts))
    return out


def _value(node: ast.expr, consts: dict[str, str]) -> object:
    """A literal, an :class:`Expr` template, or :data:`UNKNOWN`."""
    literal = _literal(node)
    if literal is not UNKNOWN:
        return literal
    var = _env_var(node)
    if var is not None:
        return Expr(f"{{$env:{var}}}")
    if isinstance(node, ast.Name):
        return Expr(consts.get(node.id, f"{{{node.id}}}"))
    template = _template(node, consts)
    return Expr(template) if template is not None else UNKNOWN


def _string_consts(body: list[ast.stmt], base: dict[str, str] | None = None) -> dict[str, str]:
    """Names bound to a static string in ``body``, on top of ``base``.

    A name bound to an environment read is the template ``{$env:VAR}``.
    """
    out: dict[str, str] = dict(base or {})
    for stmt in body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            target, value = stmt.targets[0], stmt.value
        elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
            target, value = stmt.target, stmt.value
        else:
            continue
        if isinstance(target, ast.Name):
            var = _env_var(value)
            template = f"{{$env:{var}}}" if var is not None else _template(value, out)
            if template is not None:
                out[target.id] = template
    return out


def _template(node: ast.expr, consts: dict[str, str]) -> str | None:
    """A string expression as a URL template, or ``None`` if nothing is static."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return consts.get(node.id)
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for v in node.values:
            if isinstance(v, ast.Constant):
                parts.append(str(v.value))
            elif isinstance(v, ast.FormattedValue):
                inner = v.value
                label = inner.id if isinstance(inner, ast.Name) else "?"
                var = _env_var(inner)
                if var is not None:
                    parts.append(f"{{$env:{var}}}")
                elif isinstance(inner, ast.Name) and inner.id in consts:
                    parts.append(consts[inner.id])
                else:
                    parts.append(f"{{{label}}}")
        return "".join(parts)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _template(node.left, consts)
        right = _template(node.right, consts)
        if left is None and right is None:
            return None
        return (left if left is not None else "{?}") + (right if right is not None else "{?}")
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "format"
    ):
        return _template(node.func.value, consts)
    return None


def _literal(node: ast.expr) -> object:
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError, TypeError):
        return UNKNOWN


class _Scope:
    """Call sites and response reads within one function (or module body)."""

    def __init__(
        self,
        scope: ast.Module | ast.FunctionDef | ast.AsyncFunctionDef,
        name: str | None,
        consts: dict[str, str],
        modules: dict[str, str],
        factories: dict[str, tuple[int, int]],
        dict_helpers: dict[str, tuple[ast.Dict, dict[str, str]]] | None = None,
    ) -> None:
        self.dict_helpers = dict_helpers or {}
        self.scope = scope
        self.name = name
        self.consts = consts
        self.modules = modules
        self.factories = factories
        self.nodes = list(self._own_nodes(scope))
        self.clients = self._bound(self._is_client_factory)
        self.services = self._services()
        self.dicts = self._dict_vars()

    @staticmethod
    def _own_nodes(scope: ast.AST) -> list[ast.AST]:
        """Nodes of ``scope`` not inside a nested function (those are their own scope)."""
        out: list[ast.AST] = []
        stack = list(ast.iter_child_nodes(scope))
        while stack:
            node = stack.pop()
            if isinstance(node, (*_FUNCS, ast.ClassDef)):
                continue
            out.append(node)
            stack.extend(ast.iter_child_nodes(node))
        return out

    # ── bindings ──

    def _is_client_factory(self, value: ast.expr) -> bool:
        return (
            isinstance(value, ast.Call) and _dotted(value.func, self.modules) in _CLIENT_FACTORIES
        )

    def _bound(self, test: object) -> set[str]:
        names: set[str] = set()
        for node in self.nodes:
            if isinstance(node, ast.Assign) and test(node.value):  # type: ignore[operator]
                names |= {t.id for t in node.targets if isinstance(t, ast.Name)}
            elif isinstance(node, (ast.With, ast.AsyncWith)):
                for item in node.items:
                    if test(item.context_expr) and isinstance(item.optional_vars, ast.Name):  # type: ignore[operator]
                        names.add(item.optional_vars.id)
        return names

    def _services(self) -> dict[str, tuple[tuple[str, str], tuple[str, ...]]]:
        """Name -> ``((api, version), resource chain)``.

        ``x = build("gmail", "v1", ...)`` (or a helper wrapping ``build``)
        binds a service with an empty chain; ``t = x.users().threads()``
        binds a resource, so ``t.list(...)`` is ``users.threads.list``.
        """
        out: dict[str, tuple[tuple[str, str], tuple[str, ...]]] = {}
        # ``r = svc.files().list()`` binds a request when ``r`` is later run
        # (``r.execute()``) or paged (``svc.files().list_next(r, resp)``):
        # a zero-argument call reads like a resource step, so its use decides.
        requests = self._request_names()
        ordered = sorted(
            (n for n in self.nodes if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)),
            key=lambda n: n.lineno,
        )
        for node in ordered:
            call = node.value
            assert isinstance(call, ast.Call)
            bound: tuple[tuple[str, str], tuple[str, ...]] | None = None
            api_version = self._service_call(call)
            if api_version is not None:
                bound = (api_version, ())
            else:
                chain: list[str] = []
                inner: ast.expr = call
                while (
                    isinstance(inner, ast.Call)
                    and isinstance(inner.func, ast.Attribute)
                    and not inner.args
                    and not inner.keywords
                ):
                    chain.append(inner.func.attr)
                    inner = inner.func.value
                root = self._root(inner, out)
                # ``svc.files().list().execute()`` ran a request: what it
                # binds is a response, not a resource to call methods on.
                if chain and root is not None and "execute" not in chain:
                    base, prefix = root
                    bound = (base, (*prefix, *reversed(chain)))
            if bound is not None:
                for t in node.targets:
                    if isinstance(t, ast.Name) and t.id not in requests:
                        out[t.id] = bound
        return out

    def _request_names(self) -> set[str]:
        """Names some call runs (``name.execute()``) or pages (``*_next(name, ...)``)."""
        names: set[str] = set()
        for n in self.nodes:
            if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)):
                continue
            if n.func.attr == "execute" and isinstance(n.func.value, ast.Name):
                names.add(n.func.value.id)
            elif _is_next_page(n) and isinstance(n.args[0], ast.Name):
                names.add(n.args[0].id)
        return names

    def _root(
        self, node: ast.expr, bound: dict[str, tuple[tuple[str, str], tuple[str, ...]]]
    ) -> tuple[tuple[str, str], tuple[str, ...]] | None:
        """A service or resource: a bound name, or ``build(...)`` called inline."""
        if isinstance(node, ast.Name):
            return bound.get(node.id)
        if isinstance(node, ast.Call):
            api_version = self._service_call(node)
            if api_version is not None:
                return api_version, ()
        return None

    def _service_call(self, call: ast.Call) -> tuple[str, str] | None:
        func = call.func
        if _is_build(func, self.modules):
            positions: tuple[int, int] | None = (0, 1)
        elif isinstance(func, ast.Name):
            positions = self.factories.get(func.id)
        else:
            positions = None
        if positions is None or len(call.args) <= max(positions):
            return None
        api, version = _literal(call.args[positions[0]]), _literal(call.args[positions[1]])
        if isinstance(api, str) and isinstance(version, str):
            return api, version
        return None

    def _dict_vars(self) -> dict[str, tuple[dict[str, object], bool]]:
        """Name -> (keys and literal values, complete) for dicts built in this scope."""
        out: dict[str, tuple[dict[str, object], bool]] = {}
        for node in self.nodes:
            if isinstance(node, ast.AnnAssign) and isinstance(node.value, ast.Dict):
                node = ast.Assign(targets=[node.target], value=node.value, lineno=node.lineno)
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
                keys, complete = _dict_items(node.value, self.consts)
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        prev = out.get(t.id, ({}, True))
                        out[t.id] = ({**prev[0], **keys}, prev[1] and complete)
            elif isinstance(node, ast.Assign):
                for t in node.targets:
                    if (
                        isinstance(t, ast.Subscript)
                        and isinstance(t.value, ast.Name)
                        and isinstance(t.slice, ast.Constant)
                        and isinstance(t.slice.value, str)
                    ):
                        prev = out.get(t.value.id, ({}, True))
                        out[t.value.id] = (
                            {**prev[0], t.slice.value: _value(node.value, self.consts)},
                            prev[1],
                        )
        return out

    def _kw_dict(self, node: ast.expr | None) -> tuple[dict[str, object], bool]:
        if node is None:
            return {}, True
        if isinstance(node, ast.Dict):
            return _dict_items(node, self.consts)
        if isinstance(node, ast.Name) and node.id in self.dicts:
            return self.dicts[node.id]
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in self.dict_helpers
            and not node.args
            and not node.keywords
        ):
            return _dict_items(*self.dict_helpers[node.func.id])
        return {}, False

    # ── call sites ──

    def sites(self) -> list[CallSite]:
        sites: list[CallSite] = []
        by_node: dict[int, CallSite] = {}
        # ``svc.users()`` and ``svc.users().threads()`` are resource steps of
        # the chain, not requests: only the call an ``.execute()`` (or
        # nothing) follows is one.
        steps = {
            id(n.func.value)
            for n in self.nodes
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr != "execute"
            and isinstance(n.func.value, ast.Call)
            and not n.func.value.args
            and not n.func.value.keywords
        }
        # ``t = svc.users().threads()`` binds a resource; it requests nothing.
        steps |= {
            id(n.value)
            for n in self.nodes
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id in self.services for t in n.targets)
        }
        for node in self.nodes:
            if not isinstance(node, ast.Call) or id(node) in steps:
                continue
            site = self._http(node) or self._google(node)
            if site is not None:
                sites.append(site)
                by_node[id(node)] = site
        if by_node:
            self._response_reads(by_node)
        return sites

    def _http(self, call: ast.Call) -> CallSite | None:
        func = call.func
        dotted = _dotted(func, self.modules)
        if dotted in _HTTP_FUNCTIONS:
            assert dotted is not None
            verb_name = dotted.rsplit(".", 1)[1]
        elif isinstance(func, ast.Attribute) and (
            (isinstance(func.value, ast.Name) and func.value.id in self.clients)
            or self._is_client_factory(func.value)
        ):
            verb_name = func.attr
        else:
            return None
        args = list(call.args)
        kwargs = {k.arg: k.value for k in call.keywords if k.arg}
        if verb_name == "request":
            if not args:
                return None
            verb = _literal(args.pop(0))
            method = verb.upper() if isinstance(verb, str) else None
        elif verb_name in _HTTP_VERBS:
            method = _HTTP_VERBS[verb_name]
        else:
            return None
        url_node = args[0] if args else kwargs.get("url")
        url = _template(url_node, self.consts) if url_node is not None else None
        params, complete = self._kw_dict(kwargs.get("params"))
        body, _ = self._kw_dict(kwargs.get("json"))
        headers, headers_known = self._kw_dict(kwargs.get("headers"))
        return CallSite(
            line=call.lineno,
            function=self.name,
            kind="http",
            method=method,
            url=url,
            params=params,
            params_complete=complete and not any(k.arg is None for k in call.keywords),
            body_keys=tuple(body),
            headers=headers if headers_known else None,
            auth_kwarg="auth" in kwargs,
        )

    def _google(self, call: ast.Call) -> CallSite | None:
        """``svc.a().b().method(**params)``: the chain ends on a known service name."""
        func = call.func
        if not isinstance(func, ast.Attribute):
            return None
        chain = [func.attr]
        node = func.value
        while isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            chain.append(node.func.attr)
            node = node.func.value
        root = self._root(node, self.services)
        if root is None:
            return None
        api_version, prefix = root
        full = (*prefix, *reversed(chain))
        if len(full) < 2 or chain[0] == "execute":
            return None  # a resource step, or the outer .execute() of the real call
        params: dict[str, object] = {}
        body: tuple[str, ...] = ()
        complete = True
        for k in call.keywords:
            if k.arg == "body":
                keys, _ = self._kw_dict(k.value)
                body = tuple(keys)
            elif k.arg:
                params[k.arg] = _value(k.value, self.consts)
            else:
                spread, spread_complete = self._kw_dict(k.value)
                params.update(spread)
                complete = complete and spread_complete
        return CallSite(
            line=call.lineno,
            function=self.name,
            kind="google_client",
            google_api=api_version,
            google_chain=full,
            params=params,
            params_complete=complete,
            body_keys=body,
        )

    # ── response reads ──

    def _response_reads(self, by_node: dict[int, CallSite]) -> None:
        """Bind names to responses, then record every field path read off them.

        Each read resolves against the binding in force at its own line:
        ``resp = list(...)`` then ``resp = get(...)`` checks each read
        against its own call, and a name rebound to anything else stops
        resolving at all.
        """
        # A name holding a response object (httpx) or a Google request not
        # yet run; .json() / .execute() on it gives the body.
        roots: _Bindings[CallSite] = _Bindings()
        aliases: _Bindings[tuple[CallSite, tuple[str, ...]]] = _Bindings()

        def source(expr: ast.expr) -> CallSite | None:
            # call, call.json(), call.execute(), name.json() for a response name
            if isinstance(expr, ast.Call) and id(expr) in by_node:
                return by_node[id(expr)]
            if (
                isinstance(expr, ast.Call)
                and isinstance(expr.func, ast.Attribute)
                and expr.func.attr in ("json", "execute")
            ):
                inner = expr.func.value
                if isinstance(inner, ast.Call) and id(inner) in by_node:
                    return by_node[id(inner)]
                if isinstance(inner, ast.Name):
                    return roots.get(inner.id, inner.lineno)
            return None

        def path(expr: ast.expr) -> tuple[CallSite, tuple[str, ...]] | None:
            if (
                isinstance(expr, ast.Call)
                and isinstance(expr.func, ast.Attribute)
                and expr.func.attr in ("json", "execute")
            ):
                site = source(expr)
                return (site, ()) if site is not None else None
            if isinstance(expr, ast.Name):
                return aliases.get(expr.id, expr.lineno)
            if isinstance(expr, ast.BoolOp) and isinstance(expr.op, ast.Or):
                return path(expr.values[0])
            if isinstance(expr, ast.Subscript):
                base = path(expr.value)
                if base is None:
                    return None
                key = expr.slice
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    return base[0], (*base[1], key.value)
                if isinstance(key, ast.Constant) and isinstance(key.value, int):
                    return base[0], (*base[1], "[]")
                return None
            if (
                isinstance(expr, ast.Call)
                and isinstance(expr.func, ast.Attribute)
                and expr.func.attr == "get"
                and expr.args
                and isinstance(expr.args[0], ast.Constant)
                and isinstance(expr.args[0].value, str)
            ):
                base = path(expr.func.value)
                if base is None:
                    return None
                return base[0], (*base[1], expr.args[0].value)
            return None

        comprehensions = (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)
        binders = (ast.Assign, ast.AnnAssign, ast.For, ast.AsyncFor, *comprehensions)
        ordered = sorted(
            (n for n in self.nodes if isinstance(n, binders)), key=lambda n: getattr(n, "lineno", 0)
        )
        for node in ordered:
            if isinstance(node, ast.AnnAssign):
                if node.value is None:
                    continue
                node = ast.Assign(
                    targets=[node.target],
                    value=node.value,
                    lineno=node.lineno,
                    end_lineno=node.end_lineno,
                )
            if isinstance(node, comprehensions):
                # The target is read on the comprehension's own line.
                for gen in node.generators:
                    p = path(gen.iter)
                    if isinstance(gen.target, ast.Name):
                        item = (p[0], (*p[1], "[]")) if p is not None else None
                        aliases.bind(gen.target.id, node.lineno, item)
                continue
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if not isinstance(target, ast.Name):
                    continue
                # Visible after the statement: a read in its own value is
                # of the previous binding (``x = x.get("a")``).
                visible = (node.end_lineno or node.lineno) + 1
                site = source(node.value)
                is_json = isinstance(node.value, ast.Call) and (
                    isinstance(node.value.func, ast.Attribute)
                    and node.value.func.attr in ("json", "execute")
                )
                if site is not None and not is_json:
                    roots.bind(target.id, visible, site)
                    aliases.bind(target.id, visible, None)
                else:
                    roots.bind(target.id, visible, None)
                    aliases.bind(
                        target.id, visible, (site, ()) if site is not None else path(node.value)
                    )
            elif isinstance(node, (ast.For, ast.AsyncFor)) and isinstance(node.target, ast.Name):
                p = path(node.iter)
                aliases.bind(
                    node.target.id, node.lineno, (p[0], (*p[1], "[]")) if p is not None else None
                )
        for read in self.nodes:
            if isinstance(read, (ast.Subscript, ast.Call)):
                p = path(read)
                if p is not None and p[1] and p[1][-1] != "[]":
                    reads = p[0].response_reads
                    entry = (p[1], read.lineno)
                    if entry not in reads:
                        reads.append(entry)


def _is_next_page(call: ast.Call) -> bool:
    """googleapiclient's ``<method>_next(previous_request, previous_response)``.

    Discovery method names are camelCase, so a ``_next`` suffix is only ever
    the client's generated pagination helper.
    """
    return (
        isinstance(call.func, ast.Attribute)
        and call.func.attr.endswith("_next")
        and len(call.args) == 2
        and not call.keywords
    )


class _Bindings(Generic[_T]):
    """Name -> what it is bound to, from the line each binding takes effect."""

    def __init__(self) -> None:
        self._by_name: dict[str, list[tuple[int, _T | None]]] = {}

    def bind(self, name: str, visible_from: int, value: _T | None) -> None:
        self._by_name.setdefault(name, []).append((visible_from, value))

    def get(self, name: str, line: int) -> _T | None:
        current: _T | None = None
        for visible_from, value in self._by_name.get(name, []):
            if visible_from <= line:
                current = value
        return current


def _dict_items(node: ast.Dict, consts: dict[str, str]) -> tuple[dict[str, object], bool]:
    out: dict[str, object] = {}
    complete = True
    for k, v in zip(node.keys, node.values, strict=True):
        if k is None:
            complete = False  # **spread
        elif isinstance(k, ast.Constant) and isinstance(k.value, str):
            out[k.value] = _value(v, consts)
        else:
            complete = False
    return out, complete


__all__ = ["UNKNOWN", "CallSite", "Expr", "extract"]
