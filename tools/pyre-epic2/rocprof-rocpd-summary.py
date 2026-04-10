#!/usr/bin/env python3
"""Summarize ROCProfiler SDK rocpd sqlite output for Pyre llama.cpp runs."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


def one_table(con: sqlite3.Connection, prefix: str) -> str | None:
    rows = [
        row[0]
        for row in con.execute(
            "select name from sqlite_master where type='table' and name like ? order by name",
            (f"{prefix}%",),
        )
    ]
    return rows[0] if rows else None


def us_expr(alias: str) -> str:
    return f"(({alias}.end - {alias}.start) / 1000.0)"


def print_rows(title: str, rows: list[tuple], fmt) -> None:
    print(title)
    if not rows:
        print("  <none>")
        print()
        return
    for row in rows:
        print("  " + fmt(row))
    print()


def start_filter(alias: str, start_ns: int | None) -> str:
    return f"and {alias}.start >= {start_ns}" if start_ns is not None else ""


def summarize_regions(con: sqlite3.Connection, top: int, start_ns: int | None) -> None:
    region = one_table(con, "rocpd_region_")
    string = one_table(con, "rocpd_string_")
    if not region or not string:
        return
    rows = con.execute(
        f"""
        select s.string, count(*), sum({us_expr('r')}), avg({us_expr('r')}), max({us_expr('r')})
        from {region} r
        join {string} s on s.id = r.name_id
        where 1=1 {start_filter('r', start_ns)}
        group by s.string
        order by sum(r.end - r.start) desc
        limit ?
        """,
        (top,),
    ).fetchall()
    print_rows(
        "Top API/region time",
        rows,
        lambda r: f"{r[1]:6d} calls  total={r[2]:12.3f} us  avg={r[3]:10.3f} us  max={r[4]:10.3f} us  {r[0]}",
    )


def summarize_kernels(con: sqlite3.Connection, top: int, start_ns: int | None) -> None:
    dispatch = one_table(con, "rocpd_kernel_dispatch_")
    symbol = one_table(con, "rocpd_info_kernel_symbol_")
    if not dispatch or not symbol:
        return
    rows = con.execute(
        f"""
        select coalesce(k.display_name, k.kernel_name, '<unknown>'),
               count(*),
               sum({us_expr('d')}),
               avg({us_expr('d')}),
               max({us_expr('d')}),
               max(d.workgroup_size_x),
               max(d.workgroup_size_y),
               max(d.workgroup_size_z),
               max(d.grid_size_x),
               max(d.grid_size_y),
               max(d.grid_size_z)
        from {dispatch} d
        join {symbol} k on k.id = d.kernel_id
        where 1=1 {start_filter('d', start_ns)}
        group by 1
        order by sum(d.end - d.start) desc
        limit ?
        """,
        (top,),
    ).fetchall()
    print_rows(
        "Top kernel dispatch time",
        rows,
        lambda r: (
            f"{r[1]:6d} calls  total={r[2]:10.3f} us  avg={r[3]:8.3f} us  max={r[4]:8.3f} us  "
            f"wg=({r[5]},{r[6]},{r[7]}) grid=({r[8]},{r[9]},{r[10]})  {r[0]}"
        ),
    )


def summarize_copies(con: sqlite3.Connection, top: int, start_ns: int | None) -> None:
    copies = one_table(con, "rocpd_memory_copy_")
    string = one_table(con, "rocpd_string_")
    if not copies or not string:
        return
    rows = con.execute(
        f"""
        select s.string, c.size, count(*), sum({us_expr('c')}), avg({us_expr('c')}), max({us_expr('c')})
        from {copies} c
        join {string} s on s.id = c.name_id
        where 1=1 {start_filter('c', start_ns)}
        group by s.string, c.size
        order by sum(c.end - c.start) desc
        limit ?
        """,
        (top,),
    ).fetchall()
    print_rows(
        "Top memory copies by kind and size",
        rows,
        lambda r: f"{r[2]:6d} calls  size={r[1]:12d} B  total={r[3]:10.3f} us  avg={r[4]:8.3f} us  max={r[5]:8.3f} us  {r[0]}",
    )


def summarize_allocs(con: sqlite3.Connection, top: int, start_ns: int | None) -> None:
    allocs = one_table(con, "rocpd_memory_allocate_")
    if not allocs:
        return
    rows = con.execute(
        f"""
        select coalesce(type, '<unknown>'), coalesce(level, '<unknown>'), size, count(*),
               sum({us_expr('a')}), avg({us_expr('a')}), max({us_expr('a')})
        from {allocs} a
        where 1=1 {start_filter('a', start_ns)}
        group by type, level, size
        order by sum(a.end - a.start) desc
        limit ?
        """,
        (top,),
    ).fetchall()
    print_rows(
        "Top memory allocation records",
        rows,
        lambda r: f"{r[3]:6d} calls  size={r[2]:12d} B  total={r[4]:10.3f} us  avg={r[5]:8.3f} us  max={r[6]:8.3f} us  {r[0]} {r[1]}",
    )


def resolve_start_ns(con: sqlite3.Connection, since_us: float | None, since_first_copy_size: int | None) -> int | None:
    starts = []
    if since_us is not None:
        region = one_table(con, "rocpd_region_")
        dispatch = one_table(con, "rocpd_kernel_dispatch_")
        copies = one_table(con, "rocpd_memory_copy_")
        candidates = []
        for table in (region, dispatch, copies):
            if table:
                value = con.execute(f"select min(start) from {table}").fetchone()[0]
                if value is not None:
                    candidates.append(value)
        if candidates:
            starts.append(min(candidates) + int(since_us * 1000))
    if since_first_copy_size is not None:
        copies = one_table(con, "rocpd_memory_copy_")
        if copies:
            value = con.execute(
                f"select min(start) from {copies} where size = ?",
                (since_first_copy_size,),
            ).fetchone()[0]
            if value is not None:
                starts.append(value)
    return max(starts) if starts else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("db", type=Path, help="rocprofv3 *_results.db file")
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--since-us", type=float, default=None, help="summarize only records starting this many microseconds after the first trace record")
    parser.add_argument("--since-first-copy-size", type=int, default=None, help="summarize only records at or after the first memory copy with this byte size")
    args = parser.parse_args()

    con = sqlite3.connect(args.db)
    start_ns = resolve_start_ns(con, args.since_us, args.since_first_copy_size)
    print(f"ROCProfiler database: {args.db}")
    if start_ns is not None:
        print(f"Filtered start timestamp: {start_ns}")
    print()
    summarize_regions(con, args.top, start_ns)
    summarize_kernels(con, args.top, start_ns)
    summarize_copies(con, args.top, start_ns)
    summarize_allocs(con, args.top, start_ns)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
