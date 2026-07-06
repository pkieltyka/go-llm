#!/usr/bin/env bash
# check-coverage.sh — enforce the per-package coverage floors documented in
# docs/release.md. Self-contained (go test -cover only, no external service);
# fails when any floored package drops below its floor. Floors only move up:
# after coverage-improving work, re-measure and ratchet both this table and
# the docs/release.md table to the new actuals.
set -euo pipefail

cd "$(dirname "$0")/.."

# package<space>floor(percent). Keep in sync with docs/release.md.
FLOORS="
. 81
llmtest 61
cmd/llm-cli 77
providers/anthropic 78
providers/openai 82
providers/openaicodex 69
providers/openrouter 75
providers/chatcompletions 77
providers/vllm 84
providers/ollama 100
providers/internal/responsesapi 77
providers/internal/provideroauth 71
schema 100
"

failures=0
while read -r pkg floor; do
  [ -z "$pkg" ] && continue
  out="$(go test -count=1 -cover "./$pkg" 2>&1 | tail -n 1)"
  pct="$(printf '%s\n' "$out" | grep -o 'coverage: [0-9.]*%' | grep -o '[0-9.]*' || true)"
  if [ -z "$pct" ]; then
    echo "FAIL  $pkg: could not parse coverage from: $out"
    failures=$((failures + 1))
    continue
  fi
  if awk -v pct="$pct" -v floor="$floor" 'BEGIN { exit !(pct+0 < floor+0) }'; then
    echo "FAIL  $pkg: coverage ${pct}% is below floor ${floor}%"
    failures=$((failures + 1))
  else
    echo "ok    $pkg: coverage ${pct}% (floor ${floor}%)"
  fi
done <<EOF2
$FLOORS
EOF2

if [ "$failures" -gt 0 ]; then
  echo "coverage check failed: $failures package(s) below floor" >&2
  exit 1
fi
echo "coverage check passed"
