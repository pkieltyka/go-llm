#!/bin/sh
set -eu

go_command=${1:-go}
minimum=${2:-1.26.0}
actual=$($go_command env GOVERSION)

case "$actual" in
	go[0-9]*) actual=${actual#go} ;;
	*)
		echo "unsupported Go version string: $actual" >&2
		exit 1
		;;
esac

version_at_least() {
	awk -v actual="$1" -v minimum="$2" 'BEGIN {
		n = split(actual, a, ".")
		m = split(minimum, b, ".")
		for (i = 1; i <= 3; i++) {
			av = i <= n ? a[i] + 0 : 0
			bv = i <= m ? b[i] + 0 : 0
			if (av > bv) exit 0
			if (av < bv) exit 1
		}
		exit 0
	}'
}

if ! version_at_least "$actual" "$minimum"; then
	echo "Go $minimum or newer is required; found go$actual" >&2
	exit 1
fi

echo "verified Go $actual (minimum $minimum)"
