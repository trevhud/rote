"""The DBOS-TS case of the gate message parity test.

Split from test_gate_message_parity_ts_e2e.py so each case runs in the CI
job that has its toolchain: this one needs Docker Postgres.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.test_gate_message_parity_ts_e2e import check_runtime, pytestmark  # noqa: F401


def test_dbos_ts_run_shows_the_gate_message(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    check_runtime("dbos-ts", tmp_path, capsys)
