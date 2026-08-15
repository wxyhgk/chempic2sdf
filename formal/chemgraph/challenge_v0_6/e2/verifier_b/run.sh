#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$HERE/../../../../.." && pwd)

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "用法: $0 MANIFEST OUTPUT [ARTIFACT_ROOT]" >&2
  exit 2
fi

MANIFEST=$1
OUTPUT=$2
ROOT=${3:-$(dirname -- "$MANIFEST")}

[[ -x "$HERE/verify_e2_v0_6_b" ]] || "$HERE/build.sh"
exec "$HERE/verify_e2_v0_6_b" \
  --manifest "$MANIFEST" \
  --root "$ROOT" \
  --contract-v06 "$REPO/formal/chemgraph/challenge_v0_6/prereg/p0_contract_v0_6.json" \
  --contract-v05 "$REPO/formal/chemgraph/challenge_v0_5/prereg/p0_contract_v0_5.json" \
  --output "$OUTPUT"
