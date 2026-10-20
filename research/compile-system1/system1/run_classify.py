"""Run classify.py over skills with rote's state isolated, logging spend.

usage: python run_classify.py <state-dir> <runs-dir> <mode> <model> <skill> [<skill> ...]

Sets every ROTE_*_PATH override tests/conftest.py sets (under <state-dir>),
then appends one line per call to <runs-dir>/spend.jsonl.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

state, runs, mode, model = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]
for var, rel in {
    "ROTE_MCP_CONFIG": "mcp-config/mcp.json",
    "ROTE_MCP_TOKEN_DIR": "mcp-tokens",
    "ROTE_APPS_PATH": "rote-apps/apps.json",
    "ROTE_CLOUD_CRED_PATH": "cloud-cred/cloud.json",
    "ROTE_CONFIG_PATH": "rote-config/config.yaml",
    "ROTE_PROJECT_CONFIG_PATH": "rote-config/rote.yaml",
}.items():
    os.environ[var] = str(state / rel)
for var in (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "OPENAI_API_KEY",
    "ROTE_CLOUD_TOKEN",
    "ROTE_CLOUD_URL",
):
    os.environ.pop(var, None)

sys.path.insert(0, str(Path(__file__).parent))
from classify import REPO, run  # noqa: E402

runs.mkdir(parents=True, exist_ok=True)
ledger = runs / "spend.jsonl"
for skill in sys.argv[5:]:
    tag = f"{mode}-{model}-{skill}"
    n = len(list(runs.glob(f"{tag}.*.json")))
    result = run(mode, model, REPO / "examples" / skill / "skill")
    (runs / f"{tag}.{n}.json").write_text(json.dumps(result, indent=1))
    row = {k: result[k] for k in ("mode", "model", "skill", "wall_s", "cost_usd", "prompt_chars")}
    with ledger.open("a") as f:
        f.write(json.dumps(row) + "\n")
    total = sum(json.loads(x)["cost_usd"] or 0 for x in ledger.read_text().splitlines())
    print(json.dumps({**row, "running_spend_usd": round(total, 4)}))
