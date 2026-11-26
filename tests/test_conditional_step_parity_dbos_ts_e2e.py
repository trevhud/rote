"""The DBOS-TS cases of the conditional step parity test.

Split from test_conditional_step_parity_ts_e2e.py so each case runs in the
CI job that has its toolchain: this one needs Docker Postgres.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.test_conditional_step_parity import CASES, check
from tests.test_conditional_step_parity_ts_e2e import pytestmark, run_case  # noqa: F401


@pytest.mark.parametrize("case", sorted(CASES))
def test_dbos_ts_runs_a_step_only_when_its_condition_holds(case: str, tmp_path: Path) -> None:
    check(case, run_case("dbos-ts", case, tmp_path))
