# Releasing rote to PyPI

Releases are tag-driven. Pushing a `v*` tag runs
[`.github/workflows/release.yml`](../.github/workflows/release.yml):
the fast test suite, ruff, strict mypy, and `scripts/sanity-check.sh`
gate a build job (sdist + wheel + `twine check` + a clean-venv wheel
smoke test), and a separate publish job uploads to PyPI via Trusted
Publishing (OIDC). No API tokens are stored anywhere.

## One-time setup (before the first release)

### 0. Decide the distribution name

`rote` is **taken on PyPI** — an unrelated memoization library
published there in May 2026, actively maintained, so not reclaimable
under PEP 541. As of July 2026 `rote-cli` was available (verified via
`https://pypi.org/simple/rote-cli/`, which 404s for unregistered
names — the `/project/` page always returns the SPA shell, so it's
not a reliable check).

If you go with `rote-cli`:

- change `name = "rote"` to `name = "rote-cli"` in `pyproject.toml`
- the import name (`import rote`) and console script (`rote`) stay
  as they are — distribution and import names don't have to match
- one real conflict to know about: the existing `rote` package on
  PyPI *also* installs an `import rote` module, so the two packages
  can't be installed into the same environment. Worth a note in the
  README install section when publishing.

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
   `pyproject.toml` reads it via `[tool.hatch.version]`.
2. Run the pre-PR checklist from [CLAUDE.md](../CLAUDE.md): `pytest
   tests/`, `ruff check . && ruff format .`, `mypy src/rote`,
   `./scripts/sanity-check.sh`.
3. Commit, push, and tag:

   ```sh
   git tag v0.2.0
   git push origin v0.2.0
   ```

4. CI takes it from there. The tag must match `rote.__version__`
   exactly (minus the `v` prefix) or the workflow fails before
   building anything.
5. If the `pypi` environment has required reviewers, approve the
   publish job in the Actions UI once the gates pass.

## What the wheel ships

The wheel force-includes the graduator skill bundle at
`rote/skills/rote-graduate/` (configured in `pyproject.toml`), because
`rote graduate` needs it at runtime.
`_default_graduator_skill_dir()` resolves the packaged copy first and
falls back to the repo-root `skills/` layout for editable installs.
If you move or rename the bundle, update the force-include mapping,
the resolver, and the smoke test in the release workflow together —
the workflow's clean-venv smoke test will catch a miss.
