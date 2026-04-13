# HRX Epic 2 benchmark workflow

This directory contains the Epic 2 harness for comparing CPU, Vulkan, legacy
HIP, and HRX provider modes, plus a model-free HRX kernel benchmark.

The harness prefers:

```bash
/srv/vm-shared/projects/pyre-workspace/models/Qwen3.5-35B-A3B-UD-Q4_K_L.gguf
```

when it exists. If that model is still downloading, set `LLAMA_BENCH_MODEL` to
a smaller GGUF or run only the `kernel` subcommand.

Common commands:

```bash
./tools/hrx-epic2/hrx-epic2-bench.sh build hrx
./tools/hrx-epic2/hrx-epic2-bench.sh kernel -- --ncols 2048 --nrows 128
./tools/hrx-epic2/hrx-epic2-bench.sh kernel -- --op mul_mat_vec_q6_k --ncols 2048 --nrows 4096 --cols-dst 16
./tools/hrx-epic2/hrx-epic2-bench.sh kernel -- --op mul_mat_id_q4_k_swiglu --ncols 2048 --nrows 512 --n-experts 256 --n-ids 8 --n-tokens 1
./tools/hrx-epic2/hrx-epic2-bench.sh smoke cpu hrx-fallback hrx-rmsnorm
```

Trace comparison:

```bash
./tools/hrx-epic2/hrx-trace-summary.py run-qwen --backend both --prompt 0 --gen 8
./tools/hrx-epic2/hrx-trace-summary.py summarize \
  --hrx-log /path/to/hrx-trace.log \
  --vulkan-log /path/to/vulkan-trace.log
```

The HRX side uses `GGML_HRX_TRACE_PROVIDERS=1` and summarizes provider
claims, fallback lines, hot shapes, and the backend provider summary. The Vulkan
side uses `GGML_VK_PERF_LOGGER=1` and summarizes fusion/timing labels such as
`MUL_MAT_ADD`, `MUL_MAT_ID_ADD_ID_MUL`, and `TOPK_MOE_*`.

Kernel benchmark ops: `rms_norm`, `mul_mat_vec_f32`, `mul_mat_vec_f16`,
`mul_mat_vec_bf16`, `mul_mat_vec_q4_k`, `mul_mat_vec_q5_k`,
`mul_mat_vec_q6_k`, `mul_mat_vec_q8_0`, `mul_mat_id_q4_k`,
`mul_mat_id_q4_k_mul`, and `mul_mat_id_q4_k_swiglu`.

Provider modes:

```bash
GGML_HRX_KERNEL_PROVIDER=pure_hip
GGML_HRX_KERNEL_PROVIDER=fallback
GGML_HRX_DISABLE_RMS_NORM=1
GGML_HRX_DISABLE_MUL_MAT_VEC=1
GGML_HRX_DISABLE_MUL_MAT_ID=1
GGML_HRX_TRACE_PROVIDERS=1
```

Full Qwen comparisons default to `pp512` / `tg128`:

```bash
./tools/hrx-epic2/hrx-epic2-bench.sh bench cpu vulkan hip hrx-fallback hrx-rmsnorm
```
