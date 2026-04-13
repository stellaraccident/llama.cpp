#!/usr/bin/env python3
"""Write a provenance manifest for HRX Q4 expert experiments."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path


HRX_CLAIM_RE = re.compile(r"ggml-hrx\[[^]]+\]: claim ([A-Z0-9_]+) provider=([^ ]+)(.*)")


def run_git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def llama_root() -> Path:
    return Path(__file__).resolve().parents[2]


def workspace_root(root: Path) -> Path:
    return root.parents[1]


def parse_env(items: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--env expects KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        result[key] = value
    return result


def parse_wall(path: Path | None) -> dict:
    if path is None:
        return {}
    rows = json.loads(path.read_text())
    if not rows:
        return {"path": str(path), "rows": 0}
    row = rows[0]
    return {
        "path": str(path),
        "build_commit": row.get("build_commit"),
        "avg_ts": row.get("avg_ts"),
        "stddev_ts": row.get("stddev_ts"),
        "samples_ts": row.get("samples_ts"),
        "avg_ns": row.get("avg_ns"),
        "avg_ms_per_token": (row["avg_ns"] / 1_000_000.0 / row["n_gen"]) if row.get("avg_ns") and row.get("n_gen") else None,
        "n_gen": row.get("n_gen"),
        "model_filename": row.get("model_filename"),
    }


def parse_trace(path: Path | None, top: int) -> dict:
    if path is None:
        return {}
    claims: Counter[str] = Counter()
    hot_shapes: Counter[str] = Counter()
    for line in path.read_text(errors="replace").splitlines():
        match = HRX_CLAIM_RE.search(line)
        if not match:
            continue
        op, provider, tail = match.groups()
        key = f"{op} {provider}"
        claims[key] += 1
        if op in {"MUL_MAT_ID_SWIGLU", "MUL_MAT_ID_MUL"}:
            shape = " ".join(re.findall(r"\b(?:k|rows|ids|tokens)=[^ ]+", tail))
            hot_shapes[f"{key} {shape}".strip()] += 1
    return {
        "path": str(path),
        "claims": dict(claims.most_common(top)),
        "q4_expert_shapes": dict(hot_shapes.most_common(top)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--wall-json", type=Path)
    parser.add_argument("--trace-log", type=Path)
    parser.add_argument("--rocprof-db", type=Path)
    parser.add_argument("--rocprof-summary", type=Path)
    parser.add_argument("--isa-summary", type=Path)
    parser.add_argument("--isa-json", type=Path)
    parser.add_argument("--env", action="append", default=[], help="record env toggle as KEY=VALUE")
    parser.add_argument("--note", action="append", default=[])
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--allow-stale-build-commit", action="store_true")
    args = parser.parse_args()

    root = llama_root()
    head = run_git(root, "rev-parse", "--short", "HEAD")
    manifest = {
        "name": args.name,
        "git_head": head,
        "git_branch": run_git(root, "rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(run_git(root, "status", "--porcelain")),
        "model": str(args.model) if args.model else None,
        "env": parse_env(args.env),
        "notes": args.note,
        "wall": parse_wall(args.wall_json),
        "trace": parse_trace(args.trace_log, args.top),
        "rocprof": {
            "db": str(args.rocprof_db) if args.rocprof_db else None,
            "summary": str(args.rocprof_summary) if args.rocprof_summary else None,
        },
        "isa": {
            "summary": str(args.isa_summary) if args.isa_summary else None,
            "json": str(args.isa_json) if args.isa_json else None,
        },
        "workspace_root": str(workspace_root(root)),
    }

    build_commit = manifest["wall"].get("build_commit")
    if build_commit and build_commit != head:
        message = f"wall build_commit {build_commit} does not match git HEAD {head}"
        manifest["stale_build_commit_error"] = message
        if not args.allow_stale_build_commit:
            print(f"error: {message}", file=sys.stderr)
            print("pass --allow-stale-build-commit only when documenting a known stale artifact", file=sys.stderr)
            return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
