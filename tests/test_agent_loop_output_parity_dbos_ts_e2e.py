"""The DBOS-TS case of the agent-loop declared-output parity test.

Split from test_agent_loop_output_parity_ts_e2e.py so it runs in the CI
job that has Docker Postgres.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.test_agent_loop_output_parity import (
    mcp_server,  # noqa: F401 — fixture
    messages_stub,  # noqa: F401 — fixture
)
from tests.test_agent_loop_output_parity_ts_e2e import check_ts_runtime, pytestmark  # noqa: F401


def test_dbos_ts_downstream_reads_the_loops_declared_keys(
    messages_stub: str,  # noqa: F811
    mcp_server: str,  # noqa: F811
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    check_ts_runtime("dbos-ts", messages_stub, mcp_server, tmp_path, monkeypatch)
