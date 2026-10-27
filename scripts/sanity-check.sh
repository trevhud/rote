#!/usr/bin/env bash
#
# sanity-check.sh — fail if the repository contains any of:
#   - PicnicHealth-specific identifiers (the original internal source of
#     the bundled BDR example)
#   - Plausibly-real API keys / OAuth tokens / webhook secrets
#   - Absolute home paths that would break on another machine
#   - Real customer friction claims ("BMS and Bayer pushed back")
#
# This script is the single source of truth for "is this repo safe to push".
# Run it:
#   - manually before any `git push` to a public remote
#   - in CI as a pre-merge gate
#
# It is defensive on purpose. False positives cost a second to explain;
# false negatives leak customer data.

set -euo pipefail

# Resolve the repo root regardless of where the script is invoked from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# ───── Pinned third-party data, exempt only while unmodified ─────
# The Public Suffix List lists "stord.no", a Norwegian municipality, among
# its ICANN suffixes; it is not the employer name the scan guards against.
# The file is skipped by every check below, not only the name check. That
# is safe because of the hash pin: it is skipped only while its sha256
# equals rote.vendor_specs.psl.SHA256, the upstream file exactly, so
# nothing of ours can be in it. Any edit or refresh changes the hash and
# puts the whole file back into every check.
PSL_PATH="src/rote/vendor_specs/public_suffix_list.dat"
PSL_SHA256="9b4827a9e428a0192a9ea0a6b7804a4cc653d5f39a03456d3b17447b86d78de9"
sha256_of() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | cut -d' ' -f1
    else
        shasum -a 256 "$1" | cut -d' ' -f1
    fi
}
RG_EXEMPT=()
# grep's --exclude matches a basename anywhere, so the fallback drops
# matches by this exact path instead.
GREP_SKIP=""
if [[ -f "$PSL_PATH" && "$(sha256_of "$PSL_PATH")" == "$PSL_SHA256" ]]; then
    RG_EXEMPT=(--glob "!$PSL_PATH")
    GREP_SKIP="./$PSL_PATH:"
fi

# Use ripgrep if available (much faster + respects .gitignore), fall back
# to a grep invocation that excludes the obvious cruft dirs. The fallback
# runs on GitHub's runners (no rg there), after the tests in release.yml:
# pytest's cache records test ids holding the fake secrets a test feeds the
# URL-secret check, so the tool caches must be excluded here too.
if command -v rg >/dev/null 2>&1; then
    SCANNER() {
        # $1 = pattern, $2+ = rg flags
        local pattern="$1"; shift
        rg --hidden --no-messages \
            --glob '!.git' \
            --glob '!.venv' \
            --glob '!node_modules' \
            --glob '!*.egg-info' \
            --glob '!launch' \
            --glob '!scripts/sanity-check.sh' \
            "${RG_EXEMPT[@]}" \
            "$@" \
            "$pattern" . || true
    }
else
    SCANNER() {
        local pattern="$1"; shift
        grep -rnE "$@" \
            --exclude-dir=.git \
            --exclude-dir=.venv \
            --exclude-dir=node_modules \
            --exclude-dir='*.egg-info' \
            --exclude-dir=launch \
            --exclude-dir=.pytest_cache \
            --exclude-dir=.mypy_cache \
            --exclude-dir=.ruff_cache \
            --exclude-dir=__pycache__ \
            --exclude='sanity-check.sh' \
            "$pattern" . 2>/dev/null | skip_exempt || true
    }
    skip_exempt() {
        if [[ -n "$GREP_SKIP" ]]; then
            grep -vF -- "$GREP_SKIP" || true
        else
            cat
        fi
    }
fi

failures=0

#
# Scan the repo for a pattern. Fails if any matches are found.
#
# $1 — label for the check (printed on failure)
# $2 — pattern (extended regex)
# $3+ — additional scanner flags (e.g. "-i" for case-insensitive)
#
check() {
    local label="$1"
    local pattern="$2"
    shift 2

    local results
    results="$(SCANNER "$pattern" "$@")"

    if [[ -n "$results" ]]; then
        echo "FAIL: $label"
        # Indent each line of the match for readability (pure bash).
        printf '  %s\n' "${results//$'\n'/$'\n'  }"
        echo
        failures=$((failures + 1))
    fi
}

echo "rote :: sanity-check"
echo "--------------------"

# ───── PicnicHealth / internal identifiers ─────

check "PicnicHealth name (case-insensitive)" \
    '[Pp]icnic[Hh]ealth' -i

check "picnichealth (lowercased)" \
    'picnichealth' -i

check "Internal Notion URL" \
    'notion\.so/picnic' -i

check "Internal research URL" \
    'research\.picnichealth' -i

check "Internal scripts path" \
    'ai-tools/scripts/' -i

check "Customer friction claim (BMS and Bayer)" \
    'BMS and Bayer'

# ───── Stord / current-employer identifiers ─────
# The deal-monitor / invoice-push / ops-report examples are adapted from
# real production skills; fictionalization must hold. "stord" has no
# common-English collisions ("stored" does not match), so scan broadly.

check "Stord name (case-insensitive)" \
    'stord' -i

check "Stord email" \
    '@stord\.com' -i

# ───── Personal identifiers ─────

check "PicnicHealth email" \
    '@picnichealth\.com' -i

check "Hard-coded home path" \
    '/Users/trevhud|/home/trevhud|C:\\\\Users\\\\trevhud'

check "Work-machine home path" \
    '/Users/Trevor\.Hudson'

# ───── Plausibly-real API keys / tokens / secrets ─────
# Test fixtures with obviously-fake values are allowed via an explicit
# "should-be-scrubbed" / "test-key-not-real" substring; the check below
# catches real-looking tokens that don't include any such marker.

check "Anthropic API key" \
    'sk-ant-api[0-9]{2}-[A-Za-z0-9_-]{20,}'

check "Anthropic OAuth token (non-test)" \
    'sk-ant-oat[0-9]{2}-[A-Za-z0-9_-]{20,}(?!.*(should-be-scrubbed|test-key-not-real|pass-through))' -P

check "OpenAI API key" \
    'sk-proj-[A-Za-z0-9_-]{20,}|sk-[A-Za-z0-9]{32,}'

check "GitHub classic PAT" \
    'ghp_[A-Za-z0-9]{30,}'

check "GitHub fine-grained PAT" \
    'github_pat_[A-Za-z0-9_]{30,}'

check "Slack bot token" \
    'xoxb-[A-Za-z0-9-]+'

check "Slack user token" \
    'xoxp-[A-Za-z0-9-]+'

check "AWS access key" \
    'AKIA[0-9A-Z]{16}'

check "Generic private key PEM header" \
    'BEGIN (RSA|EC|OPENSSH|PGP|DSA) PRIVATE KEY'

# ───── Result ─────

echo
if [[ "$failures" -gt 0 ]]; then
    echo "sanity-check: $failures failure(s) — do NOT push"
    exit 1
fi

echo "sanity-check: clean"
