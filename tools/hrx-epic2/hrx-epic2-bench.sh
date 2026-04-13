#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_SRC="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE_ROOT="${HRX_WORKSPACE_ROOT:-$(cd -- "${LLAMA_SRC}/../.." && pwd)}"

BUILD_ROOT="${HRX_LLAMA_BUILD_ROOT:-${WORKSPACE_ROOT}/build}"
if [[ -n "${GGML_HRX_ROCM_PATH:-}" ]]; then
    ROCM_PATH="${GGML_HRX_ROCM_PATH}"
elif [[ -d "${WORKSPACE_ROOT}/rocm" ]]; then
    ROCM_PATH="${WORKSPACE_ROOT}/rocm"
else
    echo "Missing ROCm distribution. Set GGML_HRX_ROCM_PATH or create ${WORKSPACE_ROOT}/rocm." >&2
    exit 2
fi
HRX_RUNTIME_SRC="${HRX_RUNTIME_SRC:-${WORKSPACE_ROOT}/sources/pyre-runtime}"
HRX_RUNTIME_BUILD="${HRX_RUNTIME_BUILD:-${BUILD_ROOT}/hrx-runtime-rocm713}"
HRX_RUNTIME_INSTALL="${HRX_RUNTIME_INSTALL:-${BUILD_ROOT}/hrx-runtime-rocm713-install}"

CPU_BUILD="${BUILD_ROOT}/llama-cpu"
VULKAN_BUILD="${BUILD_ROOT}/llama-vulkan"
HIP_BUILD="${BUILD_ROOT}/llama-hip-rocm713"
HRX_BUILD="${BUILD_ROOT}/llama-hrx-rocm713"

QWEN_MODEL="${WORKSPACE_ROOT}/models/Qwen3.5-35B-A3B-UD-Q4_K_L.gguf"
MODEL="${LLAMA_BENCH_MODEL:-}"
PROMPT_TOKENS="${LLAMA_BENCH_PROMPT:-}"
GEN_TOKENS="${LLAMA_BENCH_GEN:-}"
REPETITIONS="${LLAMA_BENCH_REPETITIONS:-}"
BATCH_SIZE="${LLAMA_BENCH_BATCH:-512}"
UBATCH_SIZE="${LLAMA_BENCH_UBATCH:-512}"
GPU_LAYERS="${LLAMA_BENCH_GPU_LAYERS:-}"
OUTPUT_DIR="${LLAMA_BENCH_OUTPUT_DIR:-${BUILD_ROOT}/hrx-epic2-results}"

DEFAULT_VARIANTS=(cpu vulkan hip hrx-fallback hrx-rmsnorm hrx-matvec)

usage() {
    cat <<'EOF'
Usage:
  hrx-epic2-bench.sh build [variant...]
  hrx-epic2-bench.sh smoke [variant...]
  hrx-epic2-bench.sh bench [variant...]
  hrx-epic2-bench.sh kernel [-- hrx-kernel-bench args...]

Variants:
  cpu vulkan hip hrx-fallback hrx-rmsnorm hrx-matvec hrx

Environment overrides:
  LLAMA_BENCH_MODEL=/path/to/model.gguf
  LLAMA_BENCH_PROMPT=512
  LLAMA_BENCH_GEN=128
  LLAMA_BENCH_REPETITIONS=5
  LLAMA_BENCH_BATCH=512
  LLAMA_BENCH_UBATCH=512
  LLAMA_BENCH_GPU_LAYERS=99
  LLAMA_BENCH_OUTPUT_DIR=/path/to/output
  GGML_HRX_TRACE_PROVIDERS=1
EOF
}

cmake_common_args() {
    printf '%s\n' \
        -DCMAKE_C_COMPILER=clang \
        -DCMAKE_CXX_COMPILER=clang++ \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache \
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
        -DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=lld \
        -DCMAKE_SHARED_LINKER_FLAGS=-fuse-ld=lld \
        -DCMAKE_MODULE_LINKER_FLAGS=-fuse-ld=lld \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -GNinja
}

build_hrx_runtime() {
    cmake -S "${HRX_RUNTIME_SRC}" -B "${HRX_RUNTIME_BUILD}" \
        -DHRX_IREE_SOURCE_DIR="${WORKSPACE_ROOT}/sources/iree" \
        -DCMAKE_PREFIX_PATH="${ROCM_PATH}" \
        -DHRX_BUILD_CTS=OFF \
        -DHRX_BUILD_PASSTHROUGH=OFF \
        $(cmake_common_args)
    cmake --build "${HRX_RUNTIME_BUILD}" --target hrx -j"$(nproc)"
    cmake --install "${HRX_RUNTIME_BUILD}" --prefix "${HRX_RUNTIME_INSTALL}"
}

build_llama_variant() {
    local variant="$1"
    case "${variant}" in
        cpu)
            cmake -S "${LLAMA_SRC}" -B "${CPU_BUILD}" \
                -DGGML_VULKAN=OFF -DGGML_HIP=OFF -DGGML_HRX=OFF \
                $(cmake_common_args)
            cmake --build "${CPU_BUILD}" --target llama-bench -j"$(nproc)"
            ;;
        vulkan)
            cmake -S "${LLAMA_SRC}" -B "${VULKAN_BUILD}" \
                -DGGML_VULKAN=ON -DGGML_HIP=OFF -DGGML_HRX=OFF \
                $(cmake_common_args)
            cmake --build "${VULKAN_BUILD}" --target llama-bench -j"$(nproc)"
            ;;
        hip)
            cmake -S "${LLAMA_SRC}" -B "${HIP_BUILD}" \
                -DGGML_VULKAN=OFF -DGGML_HIP=ON -DGGML_HRX=OFF \
                -DAMDGPU_TARGETS="${LLAMA_HIP_ARCH:-gfx1100}" \
                -DCMAKE_HIP_COMPILER="${ROCM_PATH}/lib/llvm/bin/clang++" \
                -DCMAKE_PREFIX_PATH="${ROCM_PATH}" \
                $(cmake_common_args)
            cmake --build "${HIP_BUILD}" --target llama-bench -j"$(nproc)"
            ;;
        hrx|hrx-fallback|hrx-rmsnorm|hrx-matvec)
            build_hrx_runtime
            cmake -S "${LLAMA_SRC}" -B "${HRX_BUILD}" \
                -DGGML_VULKAN=OFF -DGGML_HIP=OFF -DGGML_HRX=ON \
                -DGGML_HRX_ROCM_PATH="${ROCM_PATH}" \
                -DGGML_HRX_AMDGPU_TARGET="${LLAMA_HIP_ARCH:-gfx1100}" \
                -DCMAKE_PREFIX_PATH="${HRX_RUNTIME_INSTALL}" \
                $(cmake_common_args)
            cmake --build "${HRX_BUILD}" --target llama-bench test-backend-hrx hrx-kernel-bench -j"$(nproc)"
            ;;
        *)
            echo "Unknown variant: ${variant}" >&2
            exit 2
            ;;
    esac
}

default_model_path() {
    if [[ -n "${MODEL}" ]]; then
        printf '%s\n' "${MODEL}"
        return
    fi
    if [[ -f "${QWEN_MODEL}" ]]; then
        printf '%s\n' "${QWEN_MODEL}"
        return
    fi
    if [[ -f "${HRX_BUILD}/tinyllamas/stories15M-q4_0.gguf" ]]; then
        printf '%s\n' "${HRX_BUILD}/tinyllamas/stories15M-q4_0.gguf"
        return
    fi
    echo "No model found. Set LLAMA_BENCH_MODEL or run ctest --test-dir ${HRX_BUILD} -R '^test-download-model$'." >&2
    exit 1
}

variant_binary() {
    case "$1" in
        cpu) printf '%s\n' "${CPU_BUILD}/bin/llama-bench" ;;
        vulkan) printf '%s\n' "${VULKAN_BUILD}/bin/llama-bench" ;;
        hip) printf '%s\n' "${HIP_BUILD}/bin/llama-bench" ;;
        hrx|hrx-fallback|hrx-rmsnorm|hrx-matvec) printf '%s\n' "${HRX_BUILD}/bin/llama-bench" ;;
        *) return 1 ;;
    esac
}

expected_backend_label() {
    case "$1" in
        cpu) printf '%s\n' CPU ;;
        vulkan) printf '%s\n' Vulkan ;;
        hip) printf '%s\n' HIP ;;
        hrx|hrx-fallback|hrx-rmsnorm|hrx-matvec) printf '%s\n' HRX ;;
        *) return 1 ;;
    esac
}

run_variant() {
    local mode="$1"
    local variant="$2"
    local bench_bin
    bench_bin="$(variant_binary "${variant}")"
    if [[ ! -x "${bench_bin}" ]]; then
        echo "SKIP ${variant}: missing ${bench_bin}" >&2
        return 0
    fi

    local model_path
    model_path="$(default_model_path)"

    local prompt="${PROMPT_TOKENS}"
    local gen="${GEN_TOKENS}"
    local reps="${REPETITIONS}"
    local ngl="${GPU_LAYERS}"
    if [[ "${mode}" == "smoke" ]]; then
        prompt="${prompt:-16}"
        gen="${gen:-8}"
        reps="${reps:-1}"
        ngl="${ngl:-1}"
    else
        prompt="${prompt:-512}"
        gen="${gen:-128}"
        reps="${reps:-5}"
        ngl="${ngl:-99}"
    fi

    mkdir -p "${OUTPUT_DIR}"
    local output_path="${OUTPUT_DIR}/${mode}-${variant}.json"
    local -a env_prefix=()
    local -a args=(
        "${bench_bin}"
        -m "${model_path}"
        -p "${prompt}"
        -n "${gen}"
        -b "${BATCH_SIZE}"
        -ub "${UBATCH_SIZE}"
        -fa 0
        -r "${reps}"
        -o json
        --no-warmup
    )

    case "${variant}" in
        cpu)
            args+=(-ngl 0)
            ;;
        vulkan)
            args+=(-ngl "${ngl}" -dev Vulkan0)
            ;;
        hip)
            args+=(-ngl "${ngl}" -dev HIP0)
            ;;
        hrx-fallback)
            env_prefix+=(GGML_HRX_KERNEL_PROVIDER=fallback GGML_HRX_DISABLE_RMS_NORM=1 GGML_HRX_DISABLE_MUL_MAT_VEC=1 GGML_HRX_DISABLE_MUL_MAT_ID=1)
            args+=(-ngl "${ngl}" -dev HRX0)
            ;;
        hrx-rmsnorm)
            env_prefix+=(GGML_HRX_KERNEL_PROVIDER=pure_hip GGML_HRX_DISABLE_MUL_MAT_VEC=1 GGML_HRX_DISABLE_MUL_MAT_ID=1)
            args+=(-ngl "${ngl}" -dev HRX0)
            ;;
        hrx|hrx-matvec)
            env_prefix+=(GGML_HRX_KERNEL_PROVIDER=pure_hip)
            args+=(-ngl "${ngl}" -dev HRX0)
            ;;
    esac

    echo "RUN ${variant}: ${args[*]}" >&2
    env \
        LD_LIBRARY_PATH="${ROCM_PATH}/lib:${ROCM_PATH}/lib/rocm_sysdeps/lib:${HRX_RUNTIME_INSTALL}/lib64:${LD_LIBRARY_PATH:-}" \
        "${env_prefix[@]}" \
        "${args[@]}" >"${output_path}"

    local backend_label
    backend_label="$(expected_backend_label "${variant}")"
    rg -q "\"backends\": \"${backend_label}\"" "${output_path}"
    if [[ "${variant}" == hrx* ]]; then
        rg -q '"devices": "HRX0"' "${output_path}"
    fi
    echo "WROTE ${output_path}" >&2
}

run_kernel_bench() {
    local bench_bin="${HRX_BUILD}/bin/hrx-kernel-bench"
    if [[ ! -x "${bench_bin}" ]]; then
        echo "Missing ${bench_bin}; run: $0 build hrx" >&2
        exit 1
    fi
    env \
        LD_LIBRARY_PATH="${ROCM_PATH}/lib:${ROCM_PATH}/lib/rocm_sysdeps/lib:${HRX_RUNTIME_INSTALL}/lib64:${LD_LIBRARY_PATH:-}" \
        GGML_HRX_KERNEL_PROVIDER=pure_hip \
        "${bench_bin}" "$@"
}

main() {
    if [[ $# -lt 1 ]]; then
        usage
        exit 2
    fi

    local command="$1"
    shift
    local -a variants=("$@")

    case "${command}" in
        build)
            if [[ ${#variants[@]} -eq 0 ]]; then
                variants=("${DEFAULT_VARIANTS[@]}")
            fi
            for variant in "${variants[@]}"; do
                build_llama_variant "${variant}"
            done
            ;;
        smoke|bench)
            if [[ ${#variants[@]} -eq 0 ]]; then
                variants=("${DEFAULT_VARIANTS[@]}")
            fi
            for variant in "${variants[@]}"; do
                run_variant "${command}" "${variant}"
            done
            ;;
        kernel)
            if [[ ${#variants[@]} -gt 0 && "${variants[0]}" == "--" ]]; then
                variants=("${variants[@]:1}")
            fi
            run_kernel_bench "${variants[@]}"
            ;;
        -h|--help|help)
            usage
            ;;
        *)
            usage
            exit 2
            ;;
    esac
}

main "$@"
