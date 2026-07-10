#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

script=./scripts/check-coverage.sh
tmp="$(mktemp -d "${TMPDIR:-/tmp}/go-llm-coverage-test.XXXXXX")"
trap 'rm -rf "$tmp"' EXIT

cat >"$tmp/above.txt" <<'EOF'
example/pkg/file.go:10: Example 50.0%
total:                  (statements) 82.1%
EOF
cat >"$tmp/equal.txt" <<'EOF'
total: (statements) 82.0%
EOF
cat >"$tmp/below.txt" <<'EOF'
total: (statements) 81.9%
EOF
cat >"$tmp/malformed.txt" <<'EOF'
total: (statements) unavailable
EOF
cat >"$tmp/duplicate.txt" <<'EOF'
total: (statements) 82.1%
total: (statements) 82.2%
EOF

"$script" --probe-cover-output probe 82 "$tmp/above.txt" >/dev/null
"$script" --probe-cover-output probe 82 "$tmp/equal.txt" >/dev/null

for fixture in below malformed duplicate; do
  if "$script" --probe-cover-output probe 82 "$tmp/$fixture.txt" >/dev/null 2>&1; then
    echo "FAIL  probe accepted $fixture coverage output" >&2
    exit 1
  fi
done

echo "coverage script probes passed"
