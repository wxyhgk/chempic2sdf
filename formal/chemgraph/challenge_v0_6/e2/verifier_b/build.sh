#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
trap 'rm -f "$HERE/.zlib-link-check"' EXIT

command -v gcc >/dev/null || { echo "缺少 GCC，拒绝降低验证契约。" >&2; exit 1; }
printf '#include <zlib.h>\nint main(void){return zlibVersion()==0;}\n' |
  gcc -std=c11 -x c - -lz -o "$HERE/.zlib-link-check"
rm -f "$HERE/.zlib-link-check"
make -C "$HERE" clean all
