#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$HERE/../../../../.." && pwd)
SOURCE_ROOT=${1:-$REPO/formal/chemgraph/challenge_v0_6/e2/producer/result}
SOURCE_MANIFEST=${2:-$SOURCE_ROOT/manifest.json}
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

"$HERE/build.sh" >/dev/null
"$HERE/run.sh" "$SOURCE_MANIFEST" "$TMP/positive.json" "$SOURCE_ROOT"

expect_reject() {
  local mode=$1 root=$SOURCE_ROOT manifest="$TMP/$mode.json"
  if [[ $mode == differing_bytes_same_digest || $mode == png_trailing ]]; then
    root="$TMP/root-$mode"
    cp -a --reflink=auto "$SOURCE_ROOT" "$root"
    manifest="$root/manifest.tampered.json"
  fi
  "$HERE/tamper_manifest.py" "$mode" "$SOURCE_MANIFEST" "$manifest" "$root"
  if "$HERE/run.sh" "$manifest" "$TMP/$mode.report.json" "$root" >/dev/null 2>&1; then
    echo "负向测试未拒绝: $mode" >&2
    exit 1
  fi
}

for mode in tuple_id axis missing_pair extra_pair duplicate_pair wrong_cap wrong_width \
            digest same_graph corrupt_json differing_bytes_same_digest png_trailing; do
  expect_reject "$mode"
done

echo "verifier B: 正向 1 项、负向 12 项全部通过"
