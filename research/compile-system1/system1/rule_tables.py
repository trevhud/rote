"""System-1 prototype: markdown rule tables -> Python constants (EXPERIMENTAL).

Every markdown table in a skill bundle becomes a typed literal, with no
inference: the key column is the one holding backticked codes (else the
first), and each numeric column becomes a `{key: Decimal}` map. Tables with
the same header under sibling headings (one rate table per month) nest
under their heading. The output is a Python module a pure_function can
import, with a `# from:` line naming each table's source file and line.

usage: python rule_tables.py <skill-dir>
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

_SEP = re.compile(r"^\s*\|[\s:|-]+\|\s*$")
_MONEY = re.compile(r"^\$?(-?\d[\d,]*(?:\.\d+)?)%?\b")
_CODE = re.compile(r"^`([A-Za-z0-9_.-]+)`$")


@dataclass
class Table:
    file: str
    heading: str
    line: int
    header: list[str]
    rows: list[list[str]]


def tables(path: Path, label: str) -> list[Table]:
    out: list[Table] = []
    heading = ""
    lines = path.read_text().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("#"):
            heading = line.lstrip("#").strip()
        if line.strip().startswith("|") and i + 1 < len(lines) and _SEP.match(lines[i + 1]):
            header = [c.strip() for c in line.strip().strip("|").split("|")]
            rows, j = [], i + 2
            while j < len(lines) and lines[j].strip().startswith("|"):
                rows.append([c.strip() for c in lines[j].strip().strip("|").split("|")])
                j += 1
            out.append(Table(label, heading, i + 1, header, rows))
            i = j
            continue
        i += 1
    return out


def number(cell: str) -> Decimal | None:
    m = _MONEY.match(cell.strip())
    if not m:
        return None
    try:
        return Decimal(m.group(1).replace(",", ""))
    except InvalidOperation:
        return None


def ident(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").upper() or "TABLE"


def to_constants(t: Table) -> dict[str, dict[str, Decimal]]:
    """{column name: {key: value}} for every numeric column of the table."""
    code_cols = [
        c for c in range(len(t.header)) if all(_CODE.match(r[c]) for r in t.rows if c < len(r))
    ]
    key_col = code_cols[0] if code_cols else 0
    out: dict[str, dict[str, Decimal]] = {}
    for c in range(len(t.header)):
        if c == key_col:
            continue
        col: dict[str, Decimal] = {}
        for r in t.rows:
            if c >= len(r):
                continue
            n = number(r[c])
            key = _CODE.match(r[key_col]).group(1) if _CODE.match(r[key_col]) else r[key_col]
            if n is not None:
                col[key] = n
        if col and len(col) >= max(1, len(t.rows) // 2):
            out[t.header[c]] = col
    return out


def render(skill: Path) -> tuple[str, dict]:
    found: list[Table] = []
    found += tables(skill / "SKILL.md", "SKILL.md")
    if (skill / "references").is_dir():
        for ref in sorted((skill / "references").glob("*.md")):
            found += tables(ref, f"references/{ref.name}")
    groups: dict[tuple[str, tuple[str, ...]], list[Table]] = {}
    for t in found:
        groups.setdefault((t.file, tuple(t.header)), []).append(t)
    lines = [
        '"""Rule tables lifted from the skill bundle by rote System-1 (no inference)."""',
        "",
        "from decimal import Decimal",
        "",
    ]
    values: dict = {}
    for (file, header), ts in groups.items():
        nested = len(ts) > 1
        for t in ts:
            for colname, col in to_constants(t).items():
                name = ident(f"{colname}") if not nested else ident(colname)
                lines.append(f'# from: {file}:{t.line} "{t.heading}" | {" | ".join(header)}')
                if nested:
                    values.setdefault(name, {})[t.heading] = col
                else:
                    values[name] = col
    for name, val in values.items():
        lines.append(f"{name} = {val!r}")
        lines.append("")
    return "\n".join(lines), values


if __name__ == "__main__":
    text, _ = render(Path(sys.argv[1]))
    print(text)
