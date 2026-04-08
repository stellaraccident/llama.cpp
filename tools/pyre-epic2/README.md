# Pyre Epic 2 benchmark workflow

This directory contains the Epic 2 harness for comparing CPU, Vulkan, legacy
HIP, and Pyre provider modes, plus a model-free Pyre kernel benchmark.

The harness prefers:

```bash
/srv/vm-shared/projects/pyre-workspace/models/Qwen3.5-35B-A3B-UD-Q4_K_L.gguf
```

when it exists. If that model is still downloading, set `LLAMA_BENCH_MODEL` to
a smaller GGUF or run only the `kernel` subcommand.

Common commands:

```bash
./tools/pyre-epic2/pyre-epic2-bench.sh build pyre
./tools/pyre-epic2/pyre-epic2-bench.sh kernel -- --ncols 2048 --nrows 128
./tools/pyre-epic2/pyre-epic2-bench.sh smoke cpu pyre-fallback pyre-rmsnorm
```

Provider modes:

```bash
GGML_PYRE_KERNEL_PROVIDER=pure_hip
GGML_PYRE_KERNEL_PROVIDER=fallback
GGML_PYRE_DISABLE_RMS_NORM=1
GGML_PYRE_DISABLE_MUL_MAT_VEC=1
GGML_PYRE_DISABLE_MUL_MAT_ID=1
GGML_PYRE_TRACE_PROVIDERS=1
```

Full Qwen comparisons default to `pp512` / `tg128`:

```bash
./tools/pyre-epic2/pyre-epic2-bench.sh bench cpu vulkan hip pyre-fallback pyre-rmsnorm
```
