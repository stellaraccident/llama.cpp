# HRX Epic 1 benchmark and smoke workflow

This directory contains the local bring-up harness for the HRX backend in this
workspace. It is intentionally small and reproducible: one script builds the
CPU, Vulkan, HIP, and HRX variants into the workspace build roots, and the
same script runs a fixed `llama-bench` matrix for either short smoke checks or
`pp512` / `tg128` comparisons.

## Build roots

The harness uses these build directories:

```bash
/srv/vm-shared/projects/pyre-workspace/build/llama-cpu
/srv/vm-shared/projects/pyre-workspace/build/llama-vulkan
/srv/vm-shared/projects/pyre-workspace/build/llama-hip
/srv/vm-shared/projects/pyre-workspace/build/llama-hrx
/srv/vm-shared/projects/pyre-workspace/build/hrx-runtime
/srv/vm-shared/projects/pyre-workspace/build/hrx-runtime-install
```

## Exact build commands

All four llama.cpp variants use clang, ccache, lld, `RelWithDebInfo`, and Ninja.

### CPU

```bash
cmake -S /srv/vm-shared/projects/pyre-workspace/sources/llama.cpp \
  -B /srv/vm-shared/projects/pyre-workspace/build/llama-cpu \
  -DGGML_VULKAN=OFF -DGGML_HIP=OFF -DGGML_HRX=OFF \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=lld \
  -DCMAKE_SHARED_LINKER_FLAGS=-fuse-ld=lld \
  -DCMAKE_MODULE_LINKER_FLAGS=-fuse-ld=lld \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo -GNinja
cmake --build /srv/vm-shared/projects/pyre-workspace/build/llama-cpu --target llama-bench -j"$(nproc)"
```

### Vulkan

```bash
cmake -S /srv/vm-shared/projects/pyre-workspace/sources/llama.cpp \
  -B /srv/vm-shared/projects/pyre-workspace/build/llama-vulkan \
  -DGGML_VULKAN=ON -DGGML_HIP=OFF -DGGML_HRX=OFF \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=lld \
  -DCMAKE_SHARED_LINKER_FLAGS=-fuse-ld=lld \
  -DCMAKE_MODULE_LINKER_FLAGS=-fuse-ld=lld \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo -GNinja
cmake --build /srv/vm-shared/projects/pyre-workspace/build/llama-vulkan --target llama-bench -j"$(nproc)"
```

### HIP

```bash
cmake -S /srv/vm-shared/projects/pyre-workspace/sources/llama.cpp \
  -B /srv/vm-shared/projects/pyre-workspace/build/llama-hip \
  -DGGML_VULKAN=OFF -DGGML_HIP=ON -DGGML_HRX=OFF \
  -DAMDGPU_TARGETS=gfx1100 \
  -DCMAKE_HIP_COMPILER=/srv/vm-shared/projects/pyre-workspace/build/therock/dist/rocm/lib/llvm/bin/clang++ \
  -DCMAKE_PREFIX_PATH=/srv/vm-shared/projects/pyre-workspace/build/therock/dist/rocm \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=lld \
  -DCMAKE_SHARED_LINKER_FLAGS=-fuse-ld=lld \
  -DCMAKE_MODULE_LINKER_FLAGS=-fuse-ld=lld \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo -GNinja
cmake --build /srv/vm-shared/projects/pyre-workspace/build/llama-hip --target llama-bench -j"$(nproc)"
```

### HRX runtime + llama.cpp HRX backend

```bash
cmake -S /srv/vm-shared/projects/pyre-workspace/sources/pyre-runtime \
  -B /srv/vm-shared/projects/pyre-workspace/build/hrx-runtime \
  -DHRX_IREE_SOURCE_DIR=/srv/vm-shared/projects/pyre-workspace/sources/iree \
  -DCMAKE_PREFIX_PATH=/srv/vm-shared/projects/pyre-workspace/build/therock/dist/rocm \
  -DHRX_BUILD_CTS=OFF -DHRX_BUILD_PASSTHROUGH=OFF \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=lld \
  -DCMAKE_SHARED_LINKER_FLAGS=-fuse-ld=lld \
  -DCMAKE_MODULE_LINKER_FLAGS=-fuse-ld=lld \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo -GNinja
cmake --build /srv/vm-shared/projects/pyre-workspace/build/hrx-runtime --target hrx -j"$(nproc)"
cmake --install /srv/vm-shared/projects/pyre-workspace/build/hrx-runtime \
  --prefix /srv/vm-shared/projects/pyre-workspace/build/hrx-runtime-install

cmake -S /srv/vm-shared/projects/pyre-workspace/sources/llama.cpp \
  -B /srv/vm-shared/projects/pyre-workspace/build/llama-hrx \
  -DGGML_VULKAN=OFF -DGGML_HIP=OFF -DGGML_HRX=ON \
  -DGGML_HRX_ROCM_PATH=/srv/vm-shared/projects/pyre-workspace/build/therock/dist/rocm \
  -DCMAKE_PREFIX_PATH=/srv/vm-shared/projects/pyre-workspace/build/hrx-runtime-install \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=lld \
  -DCMAKE_SHARED_LINKER_FLAGS=-fuse-ld=lld \
  -DCMAKE_MODULE_LINKER_FLAGS=-fuse-ld=lld \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo -GNinja
cmake --build /srv/vm-shared/projects/pyre-workspace/build/llama-hrx --target llama-bench test-backend-hrx -j"$(nproc)"
```

## Benchmark model

For real comparisons, use a small single-GPU checkpoint such as:

```bash
export LLAMA_BENCH_MODEL=/srv/vm-shared/models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf
```

That path does not exist on this box today, so the script falls back to:

```bash
/srv/vm-shared/projects/pyre-workspace/build/llama-hrx/tinyllamas/stories15M-q4_0.gguf
```

If that tiny fixture is missing, seed it with:

```bash
ctest --test-dir /srv/vm-shared/projects/pyre-workspace/build/llama-hrx --output-on-failure -R '^test-download-model$'
```

## Command matrix

Smoke mode keeps the run short (`pp16`, `tg8`, `-r 1`, `-ngl 1` by default):

```bash
./tools/hrx-epic1/hrx-epic1-bench.sh smoke cpu hrx-fallback hrx-rmsnorm
```

Full comparison mode defaults to `pp512`, `tg128`, `-r 5`, `-ngl 99`:

```bash
export LLAMA_BENCH_MODEL=/srv/vm-shared/models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf
./tools/hrx-epic1/hrx-epic1-bench.sh bench cpu vulkan hip hrx-fallback hrx-rmsnorm
```

Equivalent one-off `llama-bench` commands:

```bash
# CPU baseline.
/srv/vm-shared/projects/pyre-workspace/build/llama-cpu/bin/llama-bench \
  -m "${LLAMA_BENCH_MODEL}" -p 512 -n 128 -b 512 -ub 512 -ngl 0 -fa 0 -r 5 -o json --no-warmup

# Vulkan baseline.
/srv/vm-shared/projects/pyre-workspace/build/llama-vulkan/bin/llama-bench \
  -m "${LLAMA_BENCH_MODEL}" -p 512 -n 128 -b 512 -ub 512 -ngl 99 -fa 0 -dev Vulkan0 -r 5 -o json --no-warmup

# Existing HIP backend baseline.
/srv/vm-shared/projects/pyre-workspace/build/llama-hip/bin/llama-bench \
  -m "${LLAMA_BENCH_MODEL}" -p 512 -n 128 -b 512 -ub 512 -ngl 99 -fa 0 -dev HIP0 -r 5 -o json --no-warmup

# HRX mostly-fallback mode.
LD_LIBRARY_PATH=/srv/vm-shared/projects/pyre-workspace/build/therock/dist/rocm/lib:/srv/vm-shared/projects/pyre-workspace/build/hrx-runtime-install/lib64:$LD_LIBRARY_PATH \
GGML_HRX_DISABLE_RMS_NORM=1 \
/srv/vm-shared/projects/pyre-workspace/build/llama-hrx/bin/llama-bench \
  -m "${LLAMA_BENCH_MODEL}" -p 512 -n 128 -b 512 -ub 512 -ngl 99 -fa 0 -dev HRX0 -r 5 -o json --no-warmup

# HRX first-provider mode (RMS_NORM direct dispatch enabled).
LD_LIBRARY_PATH=/srv/vm-shared/projects/pyre-workspace/build/therock/dist/rocm/lib:/srv/vm-shared/projects/pyre-workspace/build/hrx-runtime-install/lib64:$LD_LIBRARY_PATH \
/srv/vm-shared/projects/pyre-workspace/build/llama-hrx/bin/llama-bench \
  -m "${LLAMA_BENCH_MODEL}" -p 512 -n 128 -b 512 -ub 512 -ngl 99 -fa 0 -dev HRX0 -r 5 -o json --no-warmup
```

## Expected log cues

For HRX runs:

- `llama-bench --list-devices` should show `HRX0: ...`.
- JSON output should contain `"backends": "HRX"` and `"devices": "HRX0"`.
- `hrx-rmsnorm` mode leaves `GGML_HRX_DISABLE_RMS_NORM` unset.
- `hrx-fallback` mode sets `GGML_HRX_DISABLE_RMS_NORM=1`, so the backend keeps
  only metadata/view ops and the ggml scheduler routes compute kernels to CPU.

On this machine, the GPU is currently a single `gfx1100` device. If multiple
passed-through GPUs are added later, update `-dev ...0` and
`AMDGPU_TARGETS=gfx1100` as needed.
