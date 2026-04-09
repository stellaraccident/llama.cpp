#!/usr/bin/env python3
"""Compile Pyre HIP kernels and summarize AMDGPU ISA/resource metadata.

This is a developer inspection tool for the pure-HIP kernel catalog. It is not
intended to be a default build gate because opcode text and metadata fields can
move across ROCm/LLVM versions.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path


HOT_KERNEL_RE = re.compile(
    r"(mul_mat_vec|mul_mat_id|flash_attn|topk|rms_norm|soft_max|"
    r"gated_delta|ssm_conv|sigmoid_mul|quantize_q8_1)"
)

OPCODE_PREFIXES = (
    "v_mfma",
    "v_wmma",
    "v_dot",
    "v_fma",
    "v_mac",
    "v_mad",
    "v_pk",
    "global_load",
    "global_store",
    "flat_load",
    "flat_store",
    "s_load",
    "s_store",
    "ds_",
    "s_barrier",
    "buffer_load",
    "buffer_store",
)


@dataclass
class KernelResource:
    name: str
    vgpr_count: int | None = None
    sgpr_count: int | None = None
    vgpr_spill_count: int | None = None
    sgpr_spill_count: int | None = None
    group_segment_fixed_size: int | None = None
    private_segment_fixed_size: int | None = None
    kernarg_segment_size: int | None = None
    wavefront_size: int | None = None
    max_flat_workgroup_size: int | None = None


@dataclass
class SourceSummary:
    source: str
    catalog_names: list[str]
    bundle: str
    elf: str
    asm: str
    resources: list[KernelResource]
    opcodes: dict[str, int]
    instruction_count: int


def workspace_root() -> Path:
    return Path(__file__).resolve().parents[4]


def default_rocm_path(root: Path) -> Path:
    return Path(os.environ.get("ROCM_PATH", root / "build/therock/dist/rocm"))


def default_clang(rocm_path: Path) -> str:
    env = os.environ.get("GGML_PYRE_CLANGXX")
    if env:
        return env
    rocm_clang = rocm_path / "lib/llvm/bin/clang++"
    if rocm_clang.exists():
        return str(rocm_clang)
    return shutil.which("clang++") or "clang++"


def default_tool(rocm_path: Path, tool: str) -> str:
    rocm_tool = rocm_path / f"lib/llvm/bin/{tool}"
    if rocm_tool.exists():
        return str(rocm_tool)
    return shutil.which(tool) or tool


def load_kernel_generator(root: Path):
    path = root / "ggml/src/ggml-pyre/kernels/generate_pyre_kernels.py"
    spec = importlib.util.spec_from_file_location("generate_pyre_kernels", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def selected_sources(kernels: list[dict], pattern: str, include_all: bool) -> list[tuple[str, list[str]]]:
    regex = re.compile(pattern) if pattern else HOT_KERNEL_RE
    by_source: dict[str, list[str]] = {}
    for kernel in kernels:
        name = kernel["name"]
        source = kernel["source"]
        if include_all or regex.search(name) or regex.search(source):
            by_source.setdefault(source, []).append(name)
    return sorted(by_source.items())


def run(cmd: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def compile_source(
    clang: str,
    rocm_path: Path,
    arch: str,
    source: Path,
    bundle: Path,
    asm: Path,
) -> None:
    common = [
        clang,
        "-x",
        "hip",
        "--offload-device-only",
        f"--offload-arch={arch}",
        f"--rocm-path={rocm_path}",
        "-O3",
    ]
    result = run([*common, "-c", str(source), "-o", str(bundle)])
    if result.returncode != 0:
        raise RuntimeError(f"compile failed for {source}\n{result.stderr}")
    result = run([*common, "-S", str(source), "-o", str(asm)])
    if result.returncode != 0:
        raise RuntimeError(f"assembly compile failed for {source}\n{result.stderr}")


def unbundle(
    bundler: str,
    arch: str,
    bundle: Path,
    elf: Path,
) -> None:
    target = f"hipv4-amdgcn-amd-amdhsa--{arch}"
    result = run([
        bundler,
        "-unbundle",
        "-type=o",
        f"-targets={target}",
        f"-input={bundle}",
        f"-output={elf}",
    ])
    if result.returncode != 0:
        raise RuntimeError(f"unbundle failed for {bundle}\n{result.stderr}")


def count_opcodes(text: str) -> tuple[dict[str, int], int]:
    counts = {prefix: 0 for prefix in OPCODE_PREFIXES}
    instruction_count = 0
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith((".", ";")) or line.endswith(":"):
            continue
        if re.match(r"^[a-z_][a-z0-9_]*\b", line) is None:
            continue
        instruction_count += 1
        opcode = line.split(None, 1)[0]
        for prefix in OPCODE_PREFIXES:
            if opcode.startswith(prefix):
                counts[prefix] += 1
                break
    return {key: value for key, value in counts.items() if value}, instruction_count


def parse_int(text: str, key: str) -> int | None:
    match = re.search(rf"\.{re.escape(key)}:\s+([0-9]+)", text)
    return int(match.group(1)) if match else None


def parse_resources(readobj_text: str) -> list[KernelResource]:
    resources: list[KernelResource] = []
    for section in re.split(r"\n  - \.args:\n", readobj_text)[1:]:
        name_match = re.search(r"\.name:\s+([^\n]+)", section)
        if not name_match:
            continue
        resources.append(
            KernelResource(
                name=name_match.group(1).strip(),
                vgpr_count=parse_int(section, "vgpr_count"),
                sgpr_count=parse_int(section, "sgpr_count"),
                vgpr_spill_count=parse_int(section, "vgpr_spill_count"),
                sgpr_spill_count=parse_int(section, "sgpr_spill_count"),
                group_segment_fixed_size=parse_int(section, "group_segment_fixed_size"),
                private_segment_fixed_size=parse_int(section, "private_segment_fixed_size"),
                kernarg_segment_size=parse_int(section, "kernarg_segment_size"),
                wavefront_size=parse_int(section, "wavefront_size"),
                max_flat_workgroup_size=parse_int(section, "max_flat_workgroup_size"),
            )
        )
    return resources


def read_resources(readobj: str, elf: Path) -> list[KernelResource]:
    result = run([readobj, "--notes", str(elf)])
    if result.returncode != 0:
        raise RuntimeError(f"llvm-readobj failed for {elf}\n{result.stderr}")
    return parse_resources(result.stdout)


def summarize_source(
    source: str,
    catalog_names: list[str],
    args: argparse.Namespace,
    source_dir: Path,
    inspect_dir: Path,
) -> SourceSummary:
    stem = Path(source).stem.replace(".", "_")
    bundle = inspect_dir / f"{stem}.bundle"
    elf = inspect_dir / f"{stem}.elf"
    asm = inspect_dir / f"{stem}.s"

    compile_source(args.clang, args.rocm_path, args.arch, source_dir / source, bundle, asm)
    unbundle(args.clang_offload_bundler, args.arch, bundle, elf)
    opcodes, instruction_count = count_opcodes(asm.read_text(encoding="utf-8", errors="replace"))
    resources = read_resources(args.llvm_readobj, elf)

    return SourceSummary(
        source=source,
        catalog_names=catalog_names,
        bundle=str(bundle),
        elf=str(elf),
        asm=str(asm),
        resources=resources,
        opcodes=opcodes,
        instruction_count=instruction_count,
    )


def print_summary(summaries: list[SourceSummary]) -> None:
    for summary in summaries:
        print(f"\n{summary.source}")
        print(f"  catalog: {', '.join(summary.catalog_names)}")
        print(f"  instructions: {summary.instruction_count}")
        if summary.opcodes:
            ops = " ".join(f"{key}={value}" for key, value in sorted(summary.opcodes.items()))
            print(f"  opcodes: {ops}")
        else:
            print("  opcodes: <none matched>")
        print("  resources:")
        for resource in summary.resources:
            print(
                f"    {resource.name}: "
                f"vgpr={resource.vgpr_count} sgpr={resource.sgpr_count} "
                f"vgpr_spill={resource.vgpr_spill_count} sgpr_spill={resource.sgpr_spill_count} "
                f"lds={resource.group_segment_fixed_size} private={resource.private_segment_fixed_size} "
                f"kernarg={resource.kernarg_segment_size} wave={resource.wavefront_size} "
                f"max_wg={resource.max_flat_workgroup_size}"
            )
        print(f"  artifacts: asm={summary.asm} elf={summary.elf}")


def main() -> int:
    root = workspace_root()
    llama_root = root / "sources/llama.cpp"
    rocm_path = default_rocm_path(root)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", default="", help="regex over catalog names or source files; default selects hot kernels")
    parser.add_argument("--all", action="store_true", help="inspect every catalog source")
    parser.add_argument("--arch", default=os.environ.get("GGML_PYRE_AMDGPU_TARGET", "gfx1100"))
    parser.add_argument("--rocm-path", type=Path, default=rocm_path)
    parser.add_argument("--clang", default=default_clang(rocm_path))
    parser.add_argument("--clang-offload-bundler", default=default_tool(rocm_path, "clang-offload-bundler"))
    parser.add_argument("--llvm-readobj", default=default_tool(rocm_path, "llvm-readobj"))
    parser.add_argument("--out-dir", type=Path, default=None, help="directory for asm/ELF artifacts; default is temporary")
    parser.add_argument("--json", type=Path, default=None, help="optional JSON summary output")
    args = parser.parse_args()

    generator = load_kernel_generator(llama_root)
    source_dir = llama_root / "ggml/src/ggml-pyre/kernels"
    sources = selected_sources(generator.KERNELS, args.kernel, args.all)
    if not sources:
        print("no kernels selected", file=sys.stderr)
        return 1

    summaries: list[SourceSummary] = []
    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        for source, catalog_names in sources:
            summaries.append(summarize_source(source, catalog_names, args, source_dir, args.out_dir))
    else:
        with tempfile.TemporaryDirectory(prefix="pyre-kernel-isa-") as tmp:
            inspect_dir = Path(tmp)
            for source, catalog_names in sources:
                summaries.append(summarize_source(source, catalog_names, args, source_dir, inspect_dir))
            print_summary(summaries)
            if args.json is not None:
                args.json.write_text(json.dumps([asdict(summary) for summary in summaries], indent=2), encoding="utf-8")
            return 0

    print_summary(summaries)
    if args.json is not None:
        args.json.write_text(json.dumps([asdict(summary) for summary in summaries], indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
