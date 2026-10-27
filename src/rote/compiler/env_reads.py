"""The environment variables a node's code reads, against what its ``env:`` declares.

Every workflow runtime runs an ``extracted/`` module in the full process
environment. A compiled skill's driver runs it in an allowlisted one
(:func:`rote.skill_target.launch.child_env`), which passes a module's own
variable only when the node declares it (``Node.env``). An undeclared read
is therefore a silent behavior difference: expense-audit's report module
wrote under ``$EXPENSE_AUDIT_REPORT_DIR`` on DBOS and under ``reports/`` in
the working directory as a compiled skill (#267). So verification reads the
code and holds the declaration to it.

A read counts when its name is a literal, or a module-level name bound once
to a string literal (``REPORT_DIR_ENV = "..."``):
``os.environ["X"]``, ``os.environ.get("X")`` (and ``pop`` / ``setdefault``),
``"X" in os.environ``, ``os.getenv("X")``, with ``os`` or ``environ`` /
``getenv`` imported under any alias. The code read is the impl's module and
every ``extracted/`` module it imports, since they run in one process. Any
other use of the environment (a computed name, ``dict(os.environ)``) is
listed for review, never a failure: no declaration can name it.
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from rote.compiler.grounding import ContractQuestion
from rote.contracts import ContractFinding
from rote.env_vars import always_passed, reserved_reason
from rote.ir import Node, NodeKind, Pipeline

#: A node's code reads an environment variable its ``env:`` does not declare.
ENV_UNDECLARED = "env_undeclared"
#: A node's code reads one no declaration may name (rote's own, a credential).
ENV_RESERVED = "env_reserved"

_READ_METHODS = ("get", "pop", "setdefault")


@dataclass
class EnvReads:
    """What one module reads from the environment."""

    #: ``{name: first line}`` for each literal read.
    names: dict[str, int] = field(default_factory=dict)
    #: Lines that use the environment in a way no name can describe.
    dynamic: list[int] = field(default_factory=list)


def _module_strings(tree: ast.Module) -> dict[str, str]:
    """Module-level names bound to a string literal and bound nowhere else in the
    module (no other assignment, loop target or parameter of that name)."""
    counts: dict[str, int] = {}
    for sub in ast.walk(tree):
        if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store | ast.Del):
            counts[sub.id] = counts.get(sub.id, 0) + 1
        elif isinstance(sub, ast.arg):
            counts[sub.arg] = counts.get(sub.arg, 0) + 1
    strings: dict[str, str] = {}
    for stmt in tree.body:
        target: ast.expr | None = None
        value: ast.expr | None = None
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            target, value = stmt.targets[0], stmt.value
        elif isinstance(stmt, ast.AnnAssign):
            target, value = stmt.target, stmt.value
        if (
            isinstance(target, ast.Name)
            and isinstance(value, ast.Constant)
            and isinstance(value.value, str)
            and counts.get(target.id) == 1
        ):
            strings[target.id] = value.value
    return strings


def env_reads(source: str) -> EnvReads:
    """Every environment read in ``source`` (a module that does not parse reads nothing:
    other checks report it)."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return EnvReads()
    os_names: set[str] = set()
    environ_names: set[str] = set()
    getenv_names: set[str] = set()
    for sub in ast.walk(tree):
        if isinstance(sub, ast.Import):
            for alias in sub.names:
                if alias.name == "os" or alias.name.startswith("os."):
                    os_names.add(alias.asname or "os")
        elif isinstance(sub, ast.ImportFrom) and sub.module == "os" and not sub.level:
            for alias in sub.names:
                if alias.name == "environ":
                    environ_names.add(alias.asname or "environ")
                elif alias.name == "getenv":
                    getenv_names.add(alias.asname or "getenv")
    strings = _module_strings(tree)

    def is_environ(node: ast.AST) -> bool:
        if isinstance(node, ast.Name):
            return node.id in environ_names
        return (
            isinstance(node, ast.Attribute)
            and node.attr == "environ"
            and isinstance(node.value, ast.Name)
            and node.value.id in os_names
        )

    def is_getenv(node: ast.AST) -> bool:
        if isinstance(node, ast.Name):
            return node.id in getenv_names
        return (
            isinstance(node, ast.Attribute)
            and node.attr == "getenv"
            and isinstance(node.value, ast.Name)
            and node.value.id in os_names
        )

    def name_of(node: ast.AST | None) -> str | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name):
            return strings.get(node.id)
        return None

    found = EnvReads()

    def read(name: str | None, line: int) -> None:
        if name is None:
            found.dynamic.append(line)
        else:
            found.names.setdefault(name, line)

    # Every environ reference is accounted for exactly once: as the base of a
    # read, the target of a write, or a dynamic use.
    handled: set[int] = set()
    for sub in ast.walk(tree):
        if isinstance(sub, ast.Subscript) and is_environ(sub.value):
            handled.add(id(sub.value))
            if isinstance(sub.ctx, ast.Load):
                read(name_of(sub.slice), sub.lineno)
        elif (
            isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Attribute)
            and is_environ(sub.func.value)
        ):
            handled.add(id(sub.func.value))
            if sub.func.attr in _READ_METHODS:
                read(name_of(sub.args[0] if sub.args else None), sub.lineno)
            else:
                found.dynamic.append(sub.lineno)
        elif isinstance(sub, ast.Call) and is_getenv(sub.func):
            handled.add(id(sub.func))
            read(name_of(sub.args[0] if sub.args else None), sub.lineno)
        elif (
            isinstance(sub, ast.Compare)
            and len(sub.ops) == 1
            and isinstance(sub.ops[0], ast.In | ast.NotIn)
            and is_environ(sub.comparators[0])
        ):
            handled.add(id(sub.comparators[0]))
            read(name_of(sub.left), sub.lineno)
    for sub in ast.walk(tree):
        if (is_environ(sub) or is_getenv(sub)) and id(sub) not in handled:
            found.dynamic.append(getattr(sub, "lineno", 0))
    found.dynamic = sorted(set(found.dynamic))
    return found


def imported_modules(source: str, package: str) -> list[str]:
    """The modules of ``package`` (``extracted``) that ``source`` imports, by name."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    out: list[str] = []
    for sub in ast.walk(tree):
        if isinstance(sub, ast.Import):
            for alias in sub.names:
                head, _, rest = alias.name.partition(".")
                if head == package and rest:
                    out.append(rest.split(".")[0])
        elif isinstance(sub, ast.ImportFrom):
            module = sub.module or ""
            if sub.level == 0 and module.startswith(f"{package}."):
                out.append(module.split(".")[1])
            elif (sub.level == 0 and module == package) or (sub.level == 1 and not module):
                out += [alias.name for alias in sub.names]
            elif sub.level == 1:
                out.append(module.split(".")[0])
    return out


@dataclass(frozen=True)
class NodeEnv:
    """What one node's code reads: ``{name: (module path, line)}``, and dynamic uses."""

    names: dict[str, tuple[str, int]]
    dynamic: list[tuple[str, int]]


def node_env(node: Node, read: Callable[[str], str | None]) -> NodeEnv | None:
    """The environment ``node``'s code reads: its impl's module and every
    ``extracted/`` module that one imports. ``read(rel path)`` is a module's
    source, or ``None`` when it is missing. ``None`` for a node with no impl."""
    if node.kind not in (NodeKind.PURE_FUNCTION, NodeKind.EXTERNAL_CALL) or not node.impl:
        return None
    rel = node.impl.partition(":")[0]
    package = rel.split("/")[0]
    names: dict[str, tuple[str, int]] = {}
    dynamic: list[tuple[str, int]] = []
    queue, seen = [rel], set()
    while queue:
        path = queue.pop(0)
        if path in seen:
            continue
        seen.add(path)
        source = read(path)
        if source is None:
            continue
        reads = env_reads(source)
        for name, line in reads.names.items():
            names.setdefault(name, (path, line))
        dynamic += [(path, line) for line in reads.dynamic]
        queue += [f"{package}/{m}.py" for m in imported_modules(source, package)]
    return NodeEnv(names, dynamic)


def _reader(root: Path) -> Callable[[str], str | None]:
    def read(rel: str) -> str | None:
        try:
            return (root / rel).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return None

    return read


def declarable_reads(node: Node, root: Path) -> list[str]:
    """The names ``node``'s code reads that its ``env:`` must declare, sorted."""
    found = node_env(node, _reader(root))
    if found is None:
        return []
    return sorted(n for n in found.names if not always_passed(n) and reserved_reason(n) is None)


def env_findings(
    pipeline: Pipeline, root: Path
) -> tuple[list[ContractFinding], list[ContractQuestion]]:
    """Each literal read a node's ``env:`` does not declare (an error the agent
    repairs by declaring it), each read of a reserved name (an error: the code
    must take the value another way), and each dynamic use (a question)."""
    findings: list[ContractFinding] = []
    questions: list[ContractQuestion] = []
    read = _reader(root)
    for node in pipeline.nodes:
        found = node_env(node, read)
        if found is None:
            continue
        declared = set(node.env or ())
        for name, (path, line) in sorted(found.names.items()):
            if always_passed(name) or name in declared:
                continue
            reason = reserved_reason(name)
            if reason is not None:
                findings.append(
                    ContractFinding(
                        node_id=node.id,
                        kind=ENV_RESERVED,
                        severity="error",
                        message=(
                            f"{path}:{line} reads ${name} from the environment, which a "
                            f"compiled skill's run never passes ({reason}). Take the value as "
                            f"an input or a constant instead"
                        ),
                        path=path,
                    )
                )
                continue
            findings.append(
                ContractFinding(
                    node_id=node.id,
                    kind=ENV_UNDECLARED,
                    severity="error",
                    message=(
                        f"{path}:{line} reads ${name} from the environment, but node "
                        f"{node.id} does not declare it, so a compiled skill's run never "
                        f"passes it and the code behaves differently there. Add `env: "
                        f"[{', '.join(sorted(declared | {name}))}]` to {node.id} in "
                        f"pipeline.yaml"
                    ),
                    path=path,
                )
            )
        for path, line in found.dynamic:
            questions.append(
                ContractQuestion(
                    node.id,
                    f"{path}:{line} uses the environment by a name rote cannot read from the "
                    "code. A compiled skill's run passes only the variables a node's env: "
                    "names, so read each by a literal name and declare it",
                )
            )
    return findings, questions
