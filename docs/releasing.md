# Releasing rote to PyPI

Releases are tag-driven. Pushing a `v*` tag runs
[`.github/workflows/release.yml`](../.github/workflows/release.yml):
the fast test suite, ruff, strict mypy, and `scripts/sanity-check.sh`
gate a build job (sdist + wheel + `twine check` + a clean-venv wheel
smoke test), and a separate publish job uploads to PyPI via Trusted
Publishing (OIDC). No API tokens are stored anywhere.

`rote-cli` is **already published**. The release history is on
[PyPI](https://pypi.org/project/rote-cli/#history). The one-time setup below is
kept for the record (and in case the trusted publisher ever needs to be
re-established); day-to-day releases only need the "Cutting a release"
checklist.

## One-time setup (done — kept for reference)

### 0. Distribution name (decided: `rote-cli`)

`rote` is **taken on PyPI** — an unrelated memoization library
published there in May 2026, actively maintained, so not reclaimable
under PEP 541. The distribution is therefore named `rote-cli`
(`pyproject.toml` `name`); the import name (`import rote`) and
console script (`rote`) stay as they are — distribution and import
names don't have to match. A second console script named `rote-cli`
points at the same entry point, because `uvx rote-cli ...` runs the
command named after the distribution (`tests/test_packaging.py`).

One real conflict to know about: the existing `rote` package on
PyPI *also* installs an `import rote` module, so the two packages
can't be installed into the same environment. Worth a note in the
README install section.

### 1. PyPI: add a pending publisher

On pypi.org (logged in as the maintainer account): account sidebar →
**Publishing** → **GitHub** → fill in:

| Field | Value |
| --- | --- |
| PyPI project name | the name from step 0 |
| Owner | `trevhud` |
| Repository name | `rote` |
| Workflow filename | `release.yml` |
| Environment name | `pypi` |

A pending publisher does **not** reserve the name; it converts into a
real trusted publisher on the first successful publish. If someone
registers the name first, the pending publisher silently never
activates — another reason not to sit on step 0.

### 2. GitHub: create the `pypi` environment

Repo **Settings → Environments → New environment**, named `pypi`.
Recommended: add yourself as a required reviewer so every publish
needs a manual click after CI goes green. The publish job references
this environment; the OIDC token PyPI receives includes it, and the
pending publisher from step 1 requires it to match.

That's it. No secrets to create, rotate, or leak.

## Cutting a release

1. Bump the version in `src/rote/__init__.py` — the single source;
   `pyproject.toml` reads it via `[tool.hatch.version]`. Bump `CITATION.cff`
   to match (`tests/test_citation.py` checks it). Bump the two
   Claude Code plugin manifests to match: `plugin/.claude-plugin/plugin.json`
   and `.claude-plugin/marketplace.json` (they carry their own `version`
   and back the "install from Claude Code" path). `tests/test_plugin_manifest.py`
   fails if either drifts from `rote.__version__`.
2. Run the pre-PR checklist from [CLAUDE.md](../CLAUDE.md): `pytest
   tests/`, `ruff check . && ruff format .`, `mypy src/rote`,
   `./scripts/sanity-check.sh`.
3. Check that the published OAuth client metadata document is the one this
   release expects. `rote mcp login` offers it to servers that support
   CIMD only when the live copy matches the packaged one, so a mismatch
   silently costs every user the CIMD path:

   ```sh
   curl -fsS https://roteskills.com/oauth/rote-cli.json | python3 -m json.tool --sort-keys > /tmp/live.json
   python3 -m json.tool --sort-keys src/rote/mcp/client_metadata.json | diff - /tmp/live.json
   ```

   No output means they match. If they differ, publish the repo's
   document first (rote-cloud `marketing/oauth-client.ts`; see
   [`docs/oauth/README.md`](oauth/README.md)).
4. Commit, push, and tag:

   ```sh
   git tag v0.2.0
   git push origin v0.2.0
   ```

5. CI takes it from there. The tag must match `rote.__version__`
   exactly (minus the `v` prefix) or the workflow fails before
   building anything.
6. If the `pypi` environment has required reviewers, approve the
   publish job in the Actions UI once the gates pass.

## What the wheel ships

The wheel force-includes the compiler skill bundle at
`rote/skills/rote-compile/` (configured in `pyproject.toml`), because
`rote compile` needs it at runtime.
`_default_compiler_skill_dir()` resolves the packaged copy first and
falls back to the repo-root `skills/` layout for editable installs.
If you move or rename the bundle, update the force-include mapping,
the resolver, and the smoke test in the release workflow together —
the workflow's clean-venv smoke test will catch a miss.

The wheel also ships `rote/mcp/client_metadata.json`, the client
metadata document a login compares the published one against. It is
byte-identical to `docs/oauth/rote-cli.json`
(`tests/test_mcp_client.py::test_packaged_client_metadata_is_the_published_document`).
