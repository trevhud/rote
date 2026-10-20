"""Agent-written modules must parse on the oldest Python rote supports.

A real compile, run on a newer interpreter, wrote a backslash inside an
f-string expression: fine on 3.12+, a SyntaxError on 3.11, so the module
could not be imported where rote is supported. ``ast.parse``'s
``feature_version`` misses it (PEP 701 changed the tokenizer, not the
grammar), which is why :func:`rote.contracts.newer_syntax_problems`
exists.

The corpus below is differential across the CI matrix. Each case carries
its true 3.11 verdict. On 3.11 the test checks the verdict against the
interpreter itself, so a wrong label cannot survive CI. On 3.12+ it
checks the token-stream detector against those same verdicts.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

from rote.contracts import OLDEST_SUPPORTED_PYTHON, check_contracts, newer_syntax_problems
from rote.ir import Node, NodeKind, Pipeline

ON_OLDEST = sys.version_info[:2] <= OLDEST_SUPPORTED_PYTHON

# (source, parses on 3.11)
CORPUS: list[tuple[str, bool]] = [
    # PEP 701 relaxations: every one of these parses on 3.12+ only.
    ("x = f\"{'\\n'.join(['a', 'b'])}\"\n", False),  # the real compile's bug
    ('x = f"{"nested"}"\n', False),  # reused quote, plain string
    ('x = f"{f"{1}"}"\n', False),  # reused quote, nested f-string
    ("x = f'{1 + 1  # comment\n}'\n", False),
    ("x = f'{1 +\n 1}'\n", False),  # line break in a single-quoted f-string
    ('x = f"{ {\'a\': 1}["a"] }"\n', False),  # reused quote inside a subscript
    ('x = f"{\'a\' if True else "b"}"\n', False),
    # The grammar half: feature_version catches these.
    ("type Alias = int\n", False),
    ("def ident[T](x: T) -> T:\n    return x\n", False),
    # Shapes that look close but are valid on 3.11: the detector must stay
    # quiet on them or it would bounce healthy modules back for repair.
    ("x = f\"{'a'}\"\n", True),
    ("x = f'{\"a\"}'\n", True),
    ('x = f"""{\'a\' + "b"}"""\n', True),  # " inside triple-" is fine
    ('x = f"""{\n1 + 1\n}"""\n', True),  # line break under triple quotes
    ("x = f\"{f'{1}'}\"\n", True),  # nested f-string, other quote
    ('x = f"a\\nb{1}"\n', True),  # backslash in the literal part
    ('x = f"{1:>10}"\n', True),
    ("x = f\"{1:{'>'}{10}}\"\n", True),
    ('x = f"{{literal braces}}"\n', True),
    ('x = f"{ {1: 2}[1] }"\n', True),  # a dict display inside a field
    ('x = "\\n".join(["a"])\n', True),  # backslash outside any f-string
    ("try:\n    pass\nexcept* ValueError:\n    pass\n", True),  # 3.11 grammar
]


@pytest.mark.parametrize(("source", "parses_on_311"), CORPUS)
def test_detector_matches_the_real_3_11_parser(source: str, parses_on_311: bool) -> None:
    if ON_OLDEST:
        try:
            ast.parse(source)
            parsed = True
        except SyntaxError:
            parsed = False
        assert parsed is parses_on_311, f"corpus label is wrong for {source!r}"
    else:
        problems = newer_syntax_problems(source)
        assert (not problems) is parses_on_311, (source, problems)


def test_problems_name_the_line() -> None:
    if ON_OLDEST:
        pytest.skip("on 3.11 the interpreter's own parse reports this")
    source = "a = 1\nb = 2\nc = f\"{'\\t'.join([])}\"\n"
    [(line, message)] = newer_syntax_problems(source)
    assert line == 3
    assert "backslash" in message


def test_contract_check_reports_a_module_3_11_cannot_parse(tmp_path: Path) -> None:
    """The check runs on every agent-written module, as an error.

    An error is what bounces the module back to the agent for repair, so a
    warning here would let the module ship.
    """
    module = tmp_path / "extracted" / "report.py"
    module.parent.mkdir()
    module.write_text(
        "def render(lines: list[str]) -> str:\n    return f\"{'\\n'.join(lines)}\"\n",
        encoding="utf-8",
    )
    node = Node(
        id="render",
        kind=NodeKind.PURE_FUNCTION,
        description="d",
        impl="extracted/report.py:render",
        inputs={"lines": "pipeline.input.lines"},
    )
    pipeline = Pipeline(
        name="p",
        input={"type": "In", "required": ["lines"]},
        nodes=[node],
        edges=[],
        entry_nodes=["render"],
        exit_nodes=["render"],
    )
    findings = [f for f in check_contracts(pipeline, tmp_path) if f.severity == "error"]
    kinds = {f.kind for f in findings}
    assert kinds == ({"syntax_error"} if ON_OLDEST else {"needs_newer_python"}), findings
    assert all(f.path == "extracted/report.py" for f in findings)
