"""Score any compile output dir with `rote bench`'s own collector (no inference).

usage: python bench_row.py <out-dir holding compiled/> <skill-dir> <expected pipeline.yaml>
                           <wall-seconds> [--no-tsc]

Prints the benchmark's headline row plus the structural score, so a
parallel or hinted compile reads on the same scale as `rote bench` results.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from rote.bench.collect import collect_job
from rote.bench.report import headline

out, skill, expected, wall = (
    Path(sys.argv[1]),
    Path(sys.argv[2]),
    Path(sys.argv[3]),
    float(sys.argv[4]),
)
job = collect_job(
    out, skill, expected, exit_code=0, wall_seconds=wall, tsc="--no-tsc" not in sys.argv
)
print(
    json.dumps(
        {"headline": headline(job), "score": job.get("score"), "emit": job.get("emit")}, default=str
    )
)
