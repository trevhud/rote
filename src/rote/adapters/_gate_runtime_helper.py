# rote's gate-message renderer (rote.adapters._gate_runtime_helper), verbatim.
# Every Python workflow runtime and the compiled skill's driver use it, and
# the TS twin is GATE_MESSAGE_HELPER_TS in rote.adapters._ts_common: change a
# rule in both and run tests/test_gate_message.py with -m "slow or not slow".
# Stdlib only, imports inside the functions, and nothing from rote.


def _gate_number(value: float) -> str:
    """``value`` as JavaScript's ``JSON.stringify`` writes a number.

    Both languages pick the same shortest digits; only the layout differs
    (Python ``2.0`` and ``1e+16``, JavaScript ``2`` and
    ``10000000000000000``), so the digits come from ``repr`` and the layout
    follows ECMAScript's Number::toString.
    """
    import decimal
    import math

    if not math.isfinite(value):
        return "null"
    if value == 0:
        return "0"
    _, digit_tuple, exponent = decimal.Decimal(repr(abs(value))).as_tuple()
    assert isinstance(exponent, int)
    padded = "".join(str(d) for d in digit_tuple)
    digits = padded.rstrip("0")
    exponent += len(padded) - len(digits)
    k = len(digits)
    n = exponent + k
    if k <= n <= 21:
        text = digits + "0" * (n - k)
    elif 0 < n <= 21:
        text = digits[:n] + "." + digits[n:]
    elif -6 < n <= 0:
        text = "0." + "0" * -n + digits
    else:
        e = n - 1
        mantissa = digits if k == 1 else digits[0] + "." + digits[1:]
        text = mantissa + "e" + ("+" if e >= 0 else "-") + str(abs(e))
    return ("-" if value < 0 else "") + text


def _gate_json(value: object) -> str:
    """``value`` as JavaScript's ``JSON.stringify`` writes it, compact.

    An integer beyond 2**53 is the double JavaScript would hold, and an
    object lists its array-index keys ("0", "7") first in numeric order,
    then the rest in insertion order, as a JavaScript object does. A value
    JSON cannot carry fails, naming its type.
    """
    import json
    import re

    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, int):
        if abs(value) <= 2**53:
            return str(value)
        try:
            return _gate_number(float(value))
        except OverflowError:
            return "null"
    if isinstance(value, float):
        return _gate_number(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_gate_json(item) for item in value) + "]"
    if isinstance(value, dict):
        for key in value:
            if not isinstance(key, str):
                raise TypeError(f"object key {key!r} is not a string")
        index = sorted(
            (k for k in value if re.fullmatch("0|[1-9][0-9]*", k) and int(k) <= 2**32 - 2),
            key=int,
        )
        rest = [k for k in value if k not in set(index)]
        return (
            "{"
            + ",".join(
                json.dumps(k, ensure_ascii=False) + ":" + _gate_json(value[k]) for k in index + rest
            )
            + "}"
        )
    raise TypeError(f"{type(value).__name__} is not a JSON value")


def _gate_input(name: str, source: object, field: str) -> dict[str, object]:
    """``{name: source[field]}`` when the run has that value, else nothing.

    Spread into a gate's message values, so a value the run does not have
    (an omitted optional input, a key an upstream result lacks) is absent
    and its placeholder is kept as written, never a KeyError at the gate.
    The TS twin is ``roteGateField``.
    """
    return {name: source[field]} if isinstance(source, dict) and field in source else {}


def _render_gate_message(template: str, values: dict[str, object]) -> str:
    """A gate's message: each ``{name}`` filled from ``values``, its bound inputs.

    A string is used as is; any other value is written as JSON, the way the
    TypeScript runtimes write it (``renderGateMessage``), so one run reads
    the same on every runtime. A placeholder whose input the run left out
    (``_gate_input``) is kept as written. Then every invisible
    character (controls, format characters such as bidi overrides, line
    and paragraph separators, lone surrogates) is spelled ``\\uXXXX``, so a
    value cannot drive the terminal the message is shown on. The newline
    and the tab stay.
    """
    import re
    import unicodedata

    def fill(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in values:
            return match.group(0)
        value = values[name]
        return value if isinstance(value, str) else _gate_json(value)

    out: list[str] = []
    for ch in re.sub("\\{([A-Za-z_][A-Za-z0-9_]*)\\}", fill, template):
        if ch not in "\n\t" and unicodedata.category(ch) in ("Cc", "Cf", "Zl", "Zp", "Cs"):
            code = ord(ch)
            out.append(f"\\u{code:04x}" if code <= 0xFFFF else f"\\U{code:08x}")
        else:
            out.append(ch)
    return "".join(out)
