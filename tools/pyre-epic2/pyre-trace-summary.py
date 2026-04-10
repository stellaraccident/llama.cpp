#!/usr/bin/env python3
"""Summarize Pyre provider traces and Vulkan perf logger fusion timings."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path


PYRE_CLAIM_RE = re.compile(r"ggml-pyre\[[^]]+\]: claim ([A-Z0-9_]+) provider=([^ ]+)(.*)")
PYRE_FALLBACK_RE = re.compile(r"ggml-pyre\[[^]]+\]: fallback ([A-Z0-9_]+)(.*)")
PYRE_SUMMARY_RE = re.compile(r"provider summary (.*)$")
VULKAN_TIMING_RE = re.compile(r"^(.+?):\s+(\d+) x ([0-9.]+) us = ([0-9.]+) us")

FUSION_PREFIXES = (
    "MUL_MAT_ID_ADD_ID_MUL",
    "MUL_MAT_ID_ADD_ID",
    "MUL_MAT_ID_MUL",
    "MUL_MAT_ADD_ADD",
    "MUL_MAT_ADD",
    "RMS_NORM_MUL_ROPE_VIEW_SET_ROWS",
    "RMS_NORM_MUL_ROPE",
    "RMS_NORM_MUL",
    "ROPE_VIEW_SET_ROWS",
    "TOPK_MOE_EARLY_SOFTMAX_NORM",
    "TOPK_MOE_SIGMOID_NORM_BIAS",
    "TOPK_MOE_EARLY_SOFTMAX",
    "TOPK_MOE_LATE_SOFTMAX",
    "MULTI_ADD",
)

# Some Pyre providers intentionally cover the same graph region under a different
# name than Vulkan's fusion label. Keep this table conservative and visible so
# parity gaps do not get hidden by over-broad matching.
PYRE_EQUIVALENTS = {
    "MUL_MAT_ID_MUL": {"MUL_MAT_ID_MUL"},
    "RMS_NORM_MUL_ROPE_VIEW_SET_ROWS": {"RMS_NORM_MUL_ROPE_SET_ROWS", "RMS_NORM_MUL_ROPE"},
    "RMS_NORM_MUL_ROPE": {"RMS_NORM_MUL_ROPE"},
    "RMS_NORM_MUL": {"RMS_NORM_MUL"},
    "ROPE_VIEW_SET_ROWS": {"ROPE_SET_ROWS"},
    "TOPK_MOE_EARLY_SOFTMAX_NORM": {"TOPK_MOE", "TOPK_MOE_EARLY_SOFTMAX_NORM"},
    "TOPK_MOE_SIGMOID_NORM_BIAS": {"TOPK_MOE", "TOPK_MOE_SIGMOID_NORM_BIAS"},
    "TOPK_MOE_EARLY_SOFTMAX": {"TOPK_MOE", "TOPK_MOE_EARLY_SOFTMAX_NORM"},
    "TOPK_MOE_LATE_SOFTMAX": {"TOPK_MOE", "TOPK_MOE_EARLY_SOFTMAX_NORM"},
    "MULTI_ADD": {"ADD_ADD"},
}


def workspace_root() -> Path:
    return Path(__file__).resolve().parents[4]


def parse_shape_tail(tail: str) -> str:
    parts = []
    for key in (
        "k",
        "rows",
        "cols",
        "ne2",
        "ids",
        "tokens",
        "ncols",
        "nrows",
        "D",
        "KV",
        "N",
        "H",
        "H_KV",
        "d_conv",
        "d_inner",
    ):
        match = re.search(rf"\b{key}=([^ ]+)", tail)
        if match:
            parts.append(f"{key}={match.group(1)}")
    return " ".join(parts)


def summarize_pyre(log_paths: list[Path], top: int) -> tuple[Counter[str], list[str]]:
    claims: Counter[str] = Counter()
    op_counts: Counter[str] = Counter()
    shapes: Counter[tuple[str, str]] = Counter()
    fallbacks: Counter[str] = Counter()
    summaries: list[str] = []

    for log_path in log_paths:
        with log_path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if match := PYRE_CLAIM_RE.search(line):
                    op, provider, tail = match.groups()
                    key = f"{op} {provider}"
                    claims[key] += 1
                    op_counts[op] += 1
                    shape = parse_shape_tail(tail)
                    if shape:
                        shapes[(key, shape)] += 1
                    continue
                if match := PYRE_FALLBACK_RE.search(line):
                    op, tail = match.groups()
                    reason = tail.strip() or "unknown"
                    fallbacks[f"{op} {reason}"] += 1
                    continue
                if match := PYRE_SUMMARY_RE.search(line):
                    summaries.append(match.group(1).strip())

    print("Pyre claims")
    for key, count in claims.most_common(top):
        print(f"{count:7d}  {key}")
    if not claims:
        print("      0  <none>")

    print("\nPyre fallbacks")
    for key, count in fallbacks.most_common(top):
        print(f"{count:7d}  {key}")
    if not fallbacks:
        print("      0  <none>")

    print("\nPyre hot shapes")
    for (key, shape), count in shapes.most_common(top):
        print(f"{count:7d}  {key}  {shape}")
    if not shapes:
        print("      0  <none>")

    if summaries:
        print("\nPyre backend summaries")
        for summary in summaries[-3:]:
            print(f"  {summary}")

    return op_counts, list(fallbacks.keys())


def fusion_prefix(name: str) -> str:
    for prefix in FUSION_PREFIXES:
        if name.startswith(prefix + " ") or name == prefix:
            return prefix
    return "<unfused>"


def summarize_vulkan(log_paths: list[Path], top: int) -> Counter[str]:
    timings: Counter[str] = Counter()
    total_us: Counter[str] = Counter()
    fusions: Counter[str] = Counter()

    for log_path in log_paths:
        with log_path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                match = VULKAN_TIMING_RE.match(line.strip())
                if not match:
                    continue
                name, count_s, _avg_us_s, total_us_s = match.groups()
                count = int(count_s)
                total = float(total_us_s)
                timings[name] += count
                total_us[name] += int(total * 1000)
                fusions[fusion_prefix(name)] += count

    print("Vulkan timings")
    for name, count in timings.most_common(top):
        print(f"{count:7d}  {total_us[name] / 1000.0:12.3f} us  {name}")
    if not timings:
        print("      0  <none>")

    print("\nVulkan fusion labels")
    for name, count in fusions.most_common(top):
        print(f"{count:7d}  {name}")
    if not fusions:
        print("      0  <none>")

    return fusions


def compare_pyre_vulkan(pyre_ops: Counter[str], vulkan_fusions: Counter[str]) -> None:
    print("\nVulkan fusion labels without a direct Pyre equivalent in this trace")
    missing = []
    pyre_op_names = set(pyre_ops)
    for fusion, count in vulkan_fusions.items():
        if fusion == "<unfused>":
            continue
        equivalents = PYRE_EQUIVALENTS.get(fusion, {fusion})
        if not equivalents.intersection(pyre_op_names):
            missing.append((fusion, count, ",".join(sorted(equivalents))))
    for fusion, count, equivalents in sorted(missing, key=lambda item: (-item[1], item[0])):
        print(f"{count:7d}  {fusion}  expected-pyre={equivalents}")
    if not missing:
        print("      0  <none>")


def command_env(base: dict[str, str], binary_dir: Path, extra: dict[str, str]) -> dict[str, str]:
    env = dict(base)
    root = workspace_root()
    rocm = Path(env.get("GGML_PYRE_ROCM_PATH") or env.get("ROCM_PATH") or root / "rocm")
    pyre_install = Path(env.get("PYRE_RUNTIME_INSTALL") or root / "build/pyre-runtime-rocm713-install")
    ld_parts = [
        str(binary_dir),
        str(rocm / "lib"),
        str(rocm / "lib/rocm_sysdeps/lib"),
        str(pyre_install / "lib64"),
        env.get("LD_LIBRARY_PATH", ""),
    ]
    env["LD_LIBRARY_PATH"] = ":".join(part for part in ld_parts if part)
    env.update(extra)
    return env


def run_trace(args: argparse.Namespace) -> int:
    root = workspace_root()
    model = Path(args.model) if args.model else root / "models/Qwen3.5-35B-A3B-UD-Q4_K_L.gguf"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    backends = ["pyre", "vulkan"] if args.backend == "both" else [args.backend]
    pyre_ops = Counter()
    vulkan_fusions = Counter()
    rc = 0
    for backend in backends:
        if backend == "pyre":
            binary = root / "build/llama-pyre-rocm713/bin/llama-bench"
            dev_args = ["-ngl", str(args.gpu_layers), "-dev", "PYRE0"]
            env = command_env(os.environ, binary.parent, {
                "GGML_PYRE_KERNEL_PROVIDER": "pure_hip",
                "GGML_PYRE_TRACE_PROVIDERS": "1",
            })
        else:
            binary = root / "build/llama-vulkan/bin/llama-bench"
            dev_args = ["-ngl", str(args.gpu_layers), "-dev", "Vulkan0"]
            env = command_env(os.environ, binary.parent, {
                "GGML_VK_PERF_LOGGER": "1",
                "GGML_VK_PERF_LOGGER_FREQUENCY": "1",
            })
        if not binary.exists():
            print(f"missing benchmark binary: {binary}", file=sys.stderr)
            rc = 1
            continue

        cmd = [
            str(binary),
            "-m",
            str(model),
            "-p",
            str(args.prompt),
            "-n",
            str(args.gen),
            "-b",
            str(args.batch),
            "-ub",
            str(args.ubatch),
            "-fa",
            str(args.flash_attn),
            "-r",
            str(args.repetitions),
            "-o",
            "json",
            "--no-warmup",
            *dev_args,
        ]
        stdout_path = output_dir / f"{backend}-trace.json"
        stderr_path = output_dir / f"{backend}-trace.log"
        print("RUN", " ".join(cmd), file=sys.stderr)
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            result = subprocess.run(cmd, cwd=root, env=env, stdout=stdout, stderr=stderr, check=False)
        if result.returncode != 0:
            print(f"{backend} trace failed with exit code {result.returncode}: {stderr_path}", file=sys.stderr)
            rc = result.returncode
            continue
        print(f"WROTE {stdout_path}", file=sys.stderr)
        print(f"WROTE {stderr_path}", file=sys.stderr)
        if args.summarize:
            if backend == "pyre":
                pyre_ops, _fallbacks = summarize_pyre([stderr_path], args.top)
            else:
                vulkan_fusions = summarize_vulkan([stderr_path], args.top)
            print()
    if args.summarize and pyre_ops and vulkan_fusions:
        compare_pyre_vulkan(pyre_ops, vulkan_fusions)
    return rc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    summarize = subparsers.add_parser("summarize", help="summarize existing Pyre/Vulkan logs")
    summarize.add_argument("--pyre-log", action="append", type=Path, default=[])
    summarize.add_argument("--vulkan-log", action="append", type=Path, default=[])
    summarize.add_argument("--top", type=int, default=40)

    run = subparsers.add_parser("run-qwen", help="run a short Qwen trace and summarize it")
    run.add_argument("--backend", choices=("pyre", "vulkan", "both"), default="both")
    run.add_argument("--model")
    run.add_argument("--prompt", type=int, default=0)
    run.add_argument("--gen", type=int, default=8)
    run.add_argument("--batch", type=int, default=512)
    run.add_argument("--ubatch", type=int, default=512)
    run.add_argument("--flash-attn", type=int, default=0)
    run.add_argument("--repetitions", type=int, default=1)
    run.add_argument("--gpu-layers", type=int, default=99)
    run.add_argument("--output-dir", default=str(workspace_root() / "build/pyre-epic2-results/trace-diff"))
    run.add_argument("--top", type=int, default=40)
    run.add_argument("--summarize", action=argparse.BooleanOptionalAction, default=True)

    argv = sys.argv[1:]
    if argv and argv[0] not in {"summarize", "run-qwen", "-h", "--help"} and not argv[0].startswith("-"):
        # Backward-compatible shorthand used in older spike notes:
        #   pyre-trace-summary.py trace.log
        # Treat positional paths as Pyre provider logs.
        argv = ["summarize", "--pyre-log", argv[0], *argv[1:]]

    args = parser.parse_args(argv)
    if args.command == "run-qwen":
        return run_trace(args)

    pyre_ops = Counter()
    vulkan_fusions = Counter()
    if args.pyre_log:
        pyre_ops, _fallbacks = summarize_pyre(args.pyre_log, args.top)
    if args.pyre_log and args.vulkan_log:
        print()
    if args.vulkan_log:
        vulkan_fusions = summarize_vulkan(args.vulkan_log, args.top)
    if args.pyre_log and args.vulkan_log:
        compare_pyre_vulkan(pyre_ops, vulkan_fusions)
    if not args.pyre_log and not args.vulkan_log:
        parser.error("summarize requires --pyre-log and/or --vulkan-log")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
