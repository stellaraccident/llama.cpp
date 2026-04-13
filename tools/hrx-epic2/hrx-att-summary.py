#!/usr/bin/env python3
"""Summarize rocprofv3 ATT decoded instruction CSV output."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


OPCODE_PREFIXES = (
    "s_waitcnt",
    "global_load",
    "global_store",
    "buffer_load",
    "buffer_store",
    "flat_load",
    "flat_store",
    "ds_",
    "s_load",
    "s_store",
    "s_barrier",
    "v_mfma",
    "v_wmma",
    "v_dot",
    "v_fma",
    "v_fmac",
    "v_mul",
    "v_add",
    "v_or",
    "v_and",
    "v_lshr",
    "v_lshl",
    "v_cmp",
    "s_mul",
    "s_add",
    "s_sub",
    "s_lshr",
    "s_lshl",
    "s_and",
    "s_or",
    "s_branch",
    "s_cbranch",
)


@dataclass
class AttRow:
    code_obj: int
    vaddr: int
    instruction: str
    hitcount: int
    latency: int
    stall: int
    idle: int
    source: str


@dataclass
class Bucket:
    rows: int = 0
    hitcount: int = 0
    latency: int = 0
    stall: int = 0
    idle: int = 0


def parse_rows(path: Path) -> list[AttRow]:
    rows: list[AttRow] = []
    with path.open(newline="") as f:
        for raw in csv.DictReader(f):
            rows.append(
                AttRow(
                    code_obj=int(raw["CodeObj"]),
                    vaddr=int(raw["Vaddr"]),
                    instruction=raw["Instruction"],
                    hitcount=int(raw["Hitcount"]),
                    latency=int(raw["Latency"]),
                    stall=int(raw["Stall"]),
                    idle=int(raw["Idle"]),
                    source=raw["Source"],
                )
            )
    return rows


def opcode_bucket(instruction: str) -> str:
    if instruction.startswith(";"):
        return "label"
    opcode = instruction.split(None, 1)[0] if instruction else ""
    for prefix in OPCODE_PREFIXES:
        if opcode.startswith(prefix):
            return prefix
    return opcode or "<empty>"


def print_top(rows: list[AttRow], metric: str, top: int) -> None:
    key = {
        "stall": lambda row: row.stall,
        "latency": lambda row: row.latency,
        "idle": lambda row: row.idle,
    }[metric]
    print(f"\nTop {metric}")
    for row in sorted(rows, key=key, reverse=True)[:top]:
        value = key(row)
        if value == 0:
            break
        print(
            f"{row.vaddr:>8} {metric}={value:>10} "
            f"hit={row.hitcount:>6} lat={row.latency:>10} "
            f"stall={row.stall:>10} idle={row.idle:>10} "
            f"{row.instruction[:140]}"
        )


def print_buckets(rows: list[AttRow], top: int) -> None:
    buckets: dict[str, Bucket] = defaultdict(Bucket)
    for row in rows:
        bucket = buckets[opcode_bucket(row.instruction)]
        bucket.rows += 1
        bucket.hitcount += row.hitcount
        bucket.latency += row.latency
        bucket.stall += row.stall
        bucket.idle += row.idle

    print("\nOpcode buckets by stall")
    for name, bucket in sorted(buckets.items(), key=lambda item: item[1].stall, reverse=True)[:top]:
        print(
            f"{name:18} rows={bucket.rows:5} hit={bucket.hitcount:8} "
            f"lat={bucket.latency:12} stall={bucket.stall:12} idle={bucket.idle:12}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="stats_ui_output_agent_*_dispatch_*.csv")
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()

    rows = parse_rows(args.csv)
    labels = [row.instruction for row in rows if row.instruction.startswith(";")]
    for label in labels:
        print(label)
    print(
        f"rows={len(rows)} hit={sum(row.hitcount for row in rows)} "
        f"latency={sum(row.latency for row in rows)} "
        f"stall={sum(row.stall for row in rows)} "
        f"idle={sum(row.idle for row in rows)}"
    )
    print_top(rows, "stall", args.top)
    print_top(rows, "latency", args.top)
    print_top(rows, "idle", args.top)
    print_buckets(rows, args.top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
