#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd "$ROOT/../.." && pwd)}"
ROCM_PATH="${ROCM_PATH:-$WORKSPACE_ROOT/rocm}"
AMDGPU_TARGET="${AMDGPU_TARGET:-gfx1100}"
OUT="${OUT:-$WORKSPACE_ROOT/build/hrx-hip-kernel-bench/hrx-hip-kernel-bench}"

mkdir -p "$(dirname "$OUT")"

"$ROCM_PATH/lib/llvm/bin/clang++" \
  -x hip \
  --offload-arch="$AMDGPU_TARGET" \
  --rocm-path="$ROCM_PATH" \
  -O3 \
  -std=c++17 \
  -I"$ROOT" \
  -I"$ROOT/ggml/include" \
  -I"$ROOT/ggml/src" \
  "$ROOT/tools/hrx-epic2/hrx-hip-kernel-bench.hip.cpp" \
  -o "$OUT"

echo "$OUT"
