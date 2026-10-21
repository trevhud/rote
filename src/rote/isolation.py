"""Run rote with every per-user state path moved into one directory.

rote keeps per-user state outside the project: the app registry that
``rote emit`` appends to and ``rote mcp login`` walks, the MCP registry and
token store, the rote-cloud credential, layered config, the vendor spec
cache and fixture store, the ``rote register`` trigger registry, the
``rote eval --run`` corpus, and the price caches. Each one has a ``ROTE_*`` override. Repo
maintenance (regenerating the examples/ snapshots), the bench harness and
the test suite all need a rote that neither reads nor writes the real ones,
so the list of overrides lives here, once.

``python -m rote.isolation <rote args>`` (or ``scripts/isolated-rote``) runs
the CLI with every path under a temp dir that is deleted afterwards, or
under ``$ROTE_ISOLATED_STATE`` when that is set.
"""

from __future__ import annotations

import contextlib
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

#: Every per-user state override rote reads, relative to the state dir.
#: tests/test_isolated_rote.py derives the set from the source, so a new
#: override that is not listed here fails the suite.
STATE_PATHS: dict[str, str] = {
    "ROTE_APPS_PATH": "rote-apps/apps.json",
    "ROTE_CONFIG_PATH": "rote-config/config.yaml",
    "ROTE_PROJECT_CONFIG_PATH": "rote-config/rote.yaml",
    "ROTE_CLOUD_CRED_PATH": "cloud-cred/cloud.json",
    "ROTE_MCP_CONFIG": "mcp-config/mcp.json",
    "ROTE_MCP_TOKEN_DIR": "mcp-tokens",
    "ROTE_SPEC_CACHE_DIR": "vendor-specs",
    "ROTE_FIXTURE_DIR": "vendor-fixtures",
    "ROTE_SERVE_REGISTRY_PATH": "serve/registry.json",
    "ROTE_EVAL_CORPUS_PATH": "eval/eval-corpus.jsonl",
    "ROTE_PRICING_CACHE_DIR": "pricing-cache",
}

#: A rote-cloud login passed in the environment outranks the moved
#: credential file, so it is dropped too.
CLOUD_CREDENTIAL_ENV = ("ROTE_CLOUD_TOKEN", "ROTE_CLOUD_URL")


def isolated_env(state_dir: Path, base: Mapping[str, str] | None = None) -> dict[str, str]:
    """``base`` (default: this process's environment) with every state path moved."""
    env = dict(os.environ if base is None else base)
    for key, rel in STATE_PATHS.items():
        env[key] = str(state_dir / rel)
    for key in CLOUD_CREDENTIAL_ENV:
        env.pop(key, None)
    return env


def main(argv: Sequence[str] | None = None) -> int:
    from rote.cli import main as rote_main

    with contextlib.ExitStack() as stack:
        named = os.environ.get("ROTE_ISOLATED_STATE")
        state = (
            Path(named)
            if named
            else Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="rote-isolated-")))
        )
        env = isolated_env(state)
        os.environ.clear()
        os.environ.update(env)
        return rote_main(sys.argv[1:] if argv is None else argv)


if __name__ == "__main__":
    sys.exit(main())
