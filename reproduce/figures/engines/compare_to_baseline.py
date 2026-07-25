#!/usr/bin/env python3
"""Compare freshly regenerated figure CSVs against a preserved baseline set.

Usage:
    compare_to_baseline.py <baseline_csv_dir> <fresh_csv_dir>

For each fig<N>_*.csv present in either dir, aligns rows by their leading
non-numeric key column(s) and reports every changed cell (numeric delta or
string change), plus added/removed rows. Ends with a one-line-per-figure
summary so a glance shows which figures moved and by how much.
"""
import csv
import sys
from pathlib import Path


def load(path):
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return [], []
    return rows[0], rows[1:]


def as_num(s):
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def row_key(header, row):
    """Leading run of non-numeric columns is the row identity."""
    keys = []
    for cell in row:
        if as_num(cell) is None:
            keys.append(cell)
        else:
            break
    return tuple(keys) if keys else (row[0] if row else "",)


def index_rows(header, rows):
    idx = {}
    collision = False
    for r in rows:
        k = row_key(header, r)
        if k in idx:
            collision = True
        idx[k] = r
    return idx, collision


def compare_file(base_path, fresh_path):
    bh, brows = load(base_path) if base_path else ([], [])
    fh, frows = load(fresh_path) if fresh_path else ([], [])
    lines = []
    n_changes = 0

    if base_path is None:
        return len(frows), [f"  (no baseline — {len(frows)} fresh rows, new figure)"]
    if fresh_path is None:
        return len(brows), ["  (no fresh output — figure missing from run!)"]

    if bh != fh:
        lines.append(f"  header changed:\n    base : {bh}\n    fresh: {fh}")
        n_changes += 1

    bidx, bcol = index_rows(bh, brows)
    fidx, fcol = index_rows(fh, frows)
    if bcol or fcol:
        lines.append("  (note: duplicate row keys — alignment may be approximate)")

    keys = list(dict.fromkeys(list(bidx) + list(fidx)))
    cols = fh if fh else bh
    for k in keys:
        br, fr = bidx.get(k), fidx.get(k)
        if br is None:
            lines.append(f"  + new row {k}")
            n_changes += 1
            continue
        if fr is None:
            lines.append(f"  - removed row {k}")
            n_changes += 1
            continue
        for i, h in enumerate(cols):
            bv = as_num(br[i]) if i < len(br) else None
            fv = as_num(fr[i]) if i < len(fr) else None
            if bv is not None and fv is not None:
                if abs(fv - bv) > 1e-6:
                    lines.append(
                        f"  {k} [{h}]: {bv:g} -> {fv:g} ({fv - bv:+g})")
                    n_changes += 1
            else:
                bs = br[i] if i < len(br) else ""
                fs = fr[i] if i < len(fr) else ""
                if bs != fs:
                    lines.append(f"  {k} [{h}]: '{bs}' -> '{fs}'")
                    n_changes += 1
    return n_changes, lines


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    base_dir, fresh_dir = Path(sys.argv[1]), Path(sys.argv[2])
    names = sorted(set(
        [p.name for p in base_dir.glob("fig*.csv")]
        + [p.name for p in fresh_dir.glob("fig*.csv")]
    ))
    if not names:
        print(f"no fig*.csv found in {base_dir} or {fresh_dir}")
        sys.exit(1)

    summary = []
    for name in names:
        bp = base_dir / name
        fp = fresh_dir / name
        n, lines = compare_file(bp if bp.exists() else None,
                                fp if fp.exists() else None)
        print(f"\n=== {name} === ({n} change(s))")
        for ln in lines:
            print(ln)
        if not lines:
            print("  (identical)")
        summary.append((name, n))

    print("\n" + "=" * 50)
    print("SUMMARY (changed cells per figure)")
    print("=" * 50)
    for name, n in summary:
        flag = "unchanged" if n == 0 else f"{n} change(s)"
        print(f"  {name:<40} {flag}")


if __name__ == "__main__":
    main()
