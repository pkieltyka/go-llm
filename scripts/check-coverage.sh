#!/usr/bin/env bash
# check-coverage.sh enforces the package and owned-engine coverage floors
# documented in docs/release.md. Profiles are written only to a temporary
# directory and totals are calculated by the Go coverage tool.
set -euo pipefail

cd "$(dirname "$0")/.."

parse_coverage_percent() {
  awk '
    /^total:/ {
      value = $NF
      sub(/%$/, "", value)
      if (value !~ /^[0-9]+([.][0-9]+)?$/) {
        exit 2
      }
      total = value
      count++
    }
    END {
      if (count != 1) {
        exit 3
      }
      print total
    }
  '
}

check_coverage_output() {
  local label="$1"
  local floor="$2"
  local output="$3"
  local pct

  if ! pct="$(printf '%s\n' "$output" | parse_coverage_percent)"; then
    printf 'FAIL  %s: could not parse one total coverage value\n' "$label" >&2
    return 1
  fi
  if awk -v pct="$pct" -v floor="$floor" 'BEGIN { exit !(pct+0 < floor+0) }'; then
    printf 'FAIL  %s: coverage %s%% is below floor %s%%\n' "$label" "$pct" "$floor" >&2
    return 1
  fi
  printf 'ok    %s: coverage %s%% (floor %s%%)\n' "$label" "$pct" "$floor"
}

if [ "${1:-}" = "--probe-cover-output" ]; then
  if [ "$#" -ne 4 ]; then
    echo "usage: $0 --probe-cover-output LABEL FLOOR FILE" >&2
    exit 2
  fi
  check_coverage_output "$2" "$3" "$(cat "$4")"
  exit
fi
if [ "$#" -ne 0 ]; then
  echo "usage: $0 [--probe-cover-output LABEL FLOOR FILE]" >&2
  exit 2
fi

coverage_tmp="$(mktemp -d "${TMPDIR:-/tmp}/go-llm-coverage.XXXXXX")"
trap 'rm -rf "$coverage_tmp"' EXIT

profile_index=0
run_coverage_check() {
  local label="$1"
  local floor="$2"
  local coverpkg="$3"
  shift 3

  profile_index=$((profile_index + 1))
  local profile="$coverage_tmp/profile-$profile_index.out"
  local test_log="$coverage_tmp/test-$profile_index.log"
  local cover_output
  local -a args=(-count=1 "-coverprofile=$profile")
  if [ -n "$coverpkg" ]; then
    args+=("-coverpkg=$coverpkg")
  fi

  if ! go test "${args[@]}" "$@" >"$test_log" 2>&1; then
    printf 'FAIL  %s: go test failed\n' "$label" >&2
    sed 's/^/      /' "$test_log" >&2
    return 1
  fi
  if ! cover_output="$(go tool cover "-func=$profile" 2>&1)"; then
    printf 'FAIL  %s: go tool cover failed\n' "$label" >&2
    printf '%s\n' "$cover_output" | sed 's/^/      /' >&2
    return 1
  fi
  check_coverage_output "$label" "$floor" "$cover_output"
}

# package<space>floor(percent). Keep in sync with docs/release.md.
PACKAGE_FLOORS="
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
"

failures=0
while read -r pkg floor; do
  [ -z "$pkg" ] && continue
  if ! run_coverage_check "$pkg" "$floor" "" "./$pkg"; then
    failures=$((failures + 1))
  fi
done <<EOF
$PACKAGE_FLOORS
EOF

# These groups measure implementation ownership, not the thin public facade or
# the utility package's own tests in isolation.
if ! run_coverage_check "internal/schemajson (owned group)" 82 \
  "./internal/schemajson" ./schema ./internal/schemajson; then
  failures=$((failures + 1))
fi
if ! run_coverage_check "providers/internal/providerutil (owned group)" 76 \
  "./providers/internal/providerutil" ./providers/...; then
  failures=$((failures + 1))
fi

if [ "$failures" -gt 0 ]; then
  echo "coverage check failed: $failures floor(s) missed" >&2
  exit 1
fi
echo "coverage check passed"
