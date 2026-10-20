"""The DBOS-TS case of the ``mcp.args`` literal parity test.

Split from test_mcp_args_literal_parity_ts_e2e.py so it runs in the CI
job that has Docker Postgres.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.test_mcp_args_literal_parity import (
    echo_server,  # noqa: F401 — fixture
    token_endpoint,  # noqa: F401 — fixture
)
from tests.test_mcp_args_literal_parity_ts_e2e import check_ts_runtime, pytestmark  # noqa: F401


def test_dbos_ts_sends_literals_byte_identical(
    echo_server: str,  # noqa: F811
    token_endpoint: str,  # noqa: F811
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    check_ts_runtime("dbos-ts", echo_server, token_endpoint, tmp_path, monkeypatch)
