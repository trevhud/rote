"""The DBOS-TS case of the TS retry parity test.

Split from test_retry_parity_ts_e2e.py so each case runs in the CI job that
has its toolchain: this one needs Docker Postgres.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests.test_retry_parity import CASES
from tests.test_retry_parity_ts_e2e import check_runtime, pytestmark  # noqa: F401


@pytest.mark.parametrize(("retry", "attempts"), CASES)
def test_dbos_ts_runtime_attempts(
    retry: dict[str, Any] | None, attempts: int, tmp_path: Path
) -> None:
    check_runtime("dbos-ts", retry, attempts, tmp_path)
