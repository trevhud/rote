"""The DBOS-TS case of the TS constants parity test.

Split from test_constants_parity_ts_e2e.py so each case runs in the CI
job that has its toolchain: this one needs Docker Postgres.
"""

from __future__ import annotations

from pathlib import Path

from tests.test_constants_parity_ts_e2e import check_runtime, pytestmark  # noqa: F401


def test_dbos_ts_runtime_passes_each_nodes_own_constants(tmp_path: Path) -> None:
    check_runtime("dbos-ts", tmp_path)
