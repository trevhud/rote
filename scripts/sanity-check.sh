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

# Use ripgrep if available (much faster + respects .gitignore), fall back
# to a grep invocation that excludes the obvious cruft dirs.
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
            --exclude='sanity-check.sh' \
            "$pattern" . 2>/dev/null || true
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
