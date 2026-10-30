"""The DBOS-TS cases of the conditional gate parity test.

Split from test_conditional_gate_parity_ts_e2e.py so each case runs in the
CI job that has its toolchain: this one needs Docker Postgres.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.test_conditional_gate_parity import CASES, check
from tests.test_conditional_gate_parity_ts_e2e import pytestmark, run_case  # noqa: F401


@pytest.mark.parametrize("case", sorted(CASES))
def test_dbos_ts_asks_only_when_the_condition_holds(
    case: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run = run_case("dbos-ts", case, tmp_path)
    check(case, run, capsys.readouterr().err)
