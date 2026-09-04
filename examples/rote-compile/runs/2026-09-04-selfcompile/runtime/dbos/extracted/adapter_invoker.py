"""
Invoke the rote runtime adapter by running `rote emit` as a subprocess.

Skips execution when report_only is True, returning an empty emitted_files list.

Contract:
  Input:  pipeline_yaml_path — path to the validated pipeline.yaml
          runtime            — adapter name (dbos, temporal, cloudflare, python, etc.)
                               None means use the configured default.
          out_dir            — output directory for emitted files (None = temp dir)
          report_only        — if True, skip emission and return empty result
  Output: dict with keys:
            emitted_files — list of file paths written by the adapter
            adapter_log   — stdout+stderr from the rote emit subprocess

Raises:
  subprocess.CalledProcessError — if rote emit exits nonzero
  FileNotFoundError             — if the rote CLI is not on PATH
"""

from __future__ import annotations

import shlex
import shutil
import subprocess
from pathlib import Path


def invoke_adapter(
    pipeline_yaml_path: str,
    runtime: str | None,
    out_dir: str | None,
    report_only: bool,
) -> dict:
    """Run rote emit; return emitted file list and log. No-op when report_only."""
    if report_only:
        return {"emitted_files": [], "adapter_log": "(skipped: report_only=True)"}

    rote_bin = shutil.which("rote")
    if rote_bin is None:
        raise FileNotFoundError(
            "rote CLI not found on PATH. "
            "Install it with: pip install rote-cli"
        )

    cmd: list[str] = [rote_bin, "emit", str(pipeline_yaml_path)]
    if runtime:
        cmd += ["--runtime", runtime]
    if out_dir:
        cmd += ["--out", out_dir]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
    )

    log = (result.stdout + result.stderr).strip()

    # Parse emitted file list from the JSON summary rote emit writes to stdout.
    emitted_files: list[str] = []
    try:
        import json
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.startswith("{"):
                data = json.loads(line)
                emitted_files = data.get("files", [])
                break
    except (json.JSONDecodeError, KeyError):
        # If parsing fails, still return success with empty list.
        pass

    return {"emitted_files": emitted_files, "adapter_log": log}
