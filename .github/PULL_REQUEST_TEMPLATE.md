<!--
Keep it short. See CONTRIBUTING.md for the full expectations.
-->

## What & why

<!-- What does this change do, and why? 2-3 sentences. -->

## How verified

<!-- e.g. "pytest tests/" plus something specific to the change
     ("ran rote graduate end-to-end on BDR", "tsc --noEmit on emitted CF output"). -->

## Checklist

- [ ] `pytest tests/` passes (add `-m slow` if you touched an adapter's emitted code)
- [ ] `ruff check . && ruff format .`
- [ ] `mypy src/rote` (strict, no ignores)
- [ ] `./scripts/sanity-check.sh` exits 0
- [ ] New features land with matching tests
- [ ] If the rubric or IR changed materially: re-ran the graduator on BDR and updated the snapshot (note in the PR whether it's a *correction* or a *drift*)
- [ ] Updated `CHANGELOG.md` (Unreleased) if user-facing
