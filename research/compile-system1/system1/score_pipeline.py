"""Score an agent-compiled pipeline.yaml with the same metric as the extractor.

usage: python score_pipeline.py <skill-name> <compiled pipeline.yaml>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from score import expected_nodes, preds_from_pipeline, score  # noqa: E402

repo = Path(__file__).resolve().parents[3]
name, compiled = sys.argv[1], Path(sys.argv[2])
r = score(
    preds_from_pipeline(compiled),
    expected_nodes(repo / "examples" / name / "expected" / "pipeline.yaml"),
)
r.pop("per_section")
print(json.dumps({"skill": name, "compiled": str(compiled), **r}))
