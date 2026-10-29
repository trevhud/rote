"""The DBOS-TS case of the gate-shows parity test.

Split from test_gate_shows_parity_ts_e2e.py so it runs in the CI job that has
its toolchain: it needs Docker Postgres.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.test_gate_shows_parity import check
from tests.test_gate_shows_parity_ts_e2e import pytestmark, run_case  # noqa: F401


def test_dbos_ts_shows_the_gates_data(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    check(run_case("dbos-ts", tmp_path), capsys.readouterr().err)
