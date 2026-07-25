#!/usr/bin/env python3
"""Extract one or more kernels' SASS from a per-library megafile.

Each per-library SASS dump (e.g. `cublaslt_full_sm80.sass`) is a
concatenation of many `Fatbin elf code:` blocks; each block contains
one or more `Function : <symbol>` sections.

Given one or more target symbols, this script finds each `Function :`
block, walks back to the most recent `Fatbin elf code:` header (so the
parser sees a valid prelude), and writes a SHARED Fatbin prelude
followed by the concatenation of every requested function's body.

The lifter's parser (parse/parser_base.py) accepts multiple Function
blocks under one Fatbin header, so this produces a single .sass that
lifts to a single .ll containing every requested kernel.

Usage:
    # Single symbol (back-compat, identical to the previous behaviour)
    python scripts/utils/extract_kernel_sass.py MEGA.sass SYMBOL OUT.sass

    # Multiple symbols positionally
    python scripts/utils/extract_kernel_sass.py MEGA.sass SYM1 SYM2 ... OUT.sass

    # Comma-separated symbol list
    python scripts/utils/extract_kernel_sass.py MEGA.sass "SYM1,SYM2" OUT.sass

    # File with one symbol per line (use --out for OUT path)
    python scripts/utils/extract_kernel_sass.py MEGA.sass --symbols-file syms.txt --out OUT.sass
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


FUNC_LINE_RE = re.compile(r"^\s*Function\s*:\s*(\S+)\s*$")
FATBIN_LINE = "Fatbin elf code:"


def _extract_one(lines: List[str], symbol: str) -> Optional[Tuple[str, str]]:
    """Return (prelude_text, body_text) for a single symbol, or None.

    prelude_text is the Fatbin header + arch metadata up to the first
    Function in the symbol's containing fatbin block.  body_text is the
    text from the symbol's `Function :` line through the end of its
    body (terminated by the next `Function :` or `Fatbin elf code:`).
    """
    func_idx = None
    for i, line in enumerate(lines):
        m = FUNC_LINE_RE.match(line)
        if m and m.group(1) == symbol:
            func_idx = i
            break
    if func_idx is None:
        return None

    # Walk back to the Fatbin header for this function's elf section.
    fatbin_idx = None
    for j in range(func_idx, -1, -1):
        if lines[j].rstrip("\n") == FATBIN_LINE:
            fatbin_idx = j
            break
    if fatbin_idx is None:
        fatbin_idx = 0

    # The Fatbin block can contain MANY functions sharing one header;
    # the prelude is everything before the FIRST Function in that block.
    prelude_end = func_idx
    for j in range(fatbin_idx, func_idx):
        if FUNC_LINE_RE.match(lines[j]):
            prelude_end = j
            break

    # Walk forward to find where this function ends.  The body ends at
    # the next `Function :` line OR at the next Fatbin block.
    end_idx = len(lines)
    for k in range(func_idx + 1, len(lines)):
        if FUNC_LINE_RE.match(lines[k]) or lines[k].rstrip("\n") == FATBIN_LINE:
            end_idx = k
            break

    prelude_text = "".join(lines[fatbin_idx:prelude_end])
    body_text = "".join(lines[func_idx:end_idx])
    return prelude_text, body_text


def extract(mega_path: Path, symbols: Iterable[str], out_path: Path) -> bool:
    """Extract one or more Function blocks into a single SASS file.

    The first found symbol's surrounding Fatbin prelude is used as the
    shared prelude; each subsequent symbol contributes only its body.
    Returns True if at least one symbol was successfully extracted AND
    every requested symbol was found.
    """
    symbols = list(symbols)
    if not symbols:
        print("ERROR: no symbols requested", file=sys.stderr)
        return False

    lines = mega_path.read_text().splitlines(keepends=True)

    prelude: Optional[str] = None
    bodies: List[str] = []
    missing: List[str] = []
    line_counts: List[Tuple[str, int]] = []

    for sym in symbols:
        result = _extract_one(lines, sym)
        if result is None:
            missing.append(sym)
            continue
        sym_prelude, sym_body = result
        if prelude is None:
            prelude = sym_prelude
        bodies.append(sym_body)
        line_counts.append((sym, sym_body.count("\n")))

    if missing:
        for s in missing:
            print(f"ERROR: symbol {s!r} not found in {mega_path.name}",
                  file=sys.stderr)
        return False

    if prelude is None or not bodies:
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(prelude + "".join(bodies))
    prelude_lines = prelude.count("\n")
    print(f"wrote {out_path}")
    print(f"  prelude: {prelude_lines} lines")
    for sym, nlines in line_counts:
        print(f"  body[{sym[:50]}]: {nlines} lines")
    return True


def _parse_symbols(args) -> List[str]:
    """Resolve the symbols requested on the command line.

    Supports three input shapes (in priority order):
      * --symbols-file FILE  -> one symbol per line (blank/comment lines
                                stripped); positional args ignored.
      * positional symbols   -> one or more SYMBOL args between MEGA and
                                OUT.  A single SYMBOL arg may itself be a
                                comma-separated list, which is split.
    """
    if args.symbols_file is not None:
        syms: List[str] = []
        for raw in Path(args.symbols_file).read_text().splitlines():
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            syms.append(s)
        return syms

    syms: List[str] = []
    for tok in args.symbols:
        for piece in tok.split(","):
            piece = piece.strip()
            if piece:
                syms.append(piece)
    return syms


def main():
    ap = argparse.ArgumentParser(
        description="Extract one or more kernels' SASS from a megafile.")
    ap.add_argument("mega", type=Path, help="Per-library megafile.sass")
    ap.add_argument("symbols", nargs="*",
                    help="One or more mangled kernel symbols (or a single "
                         "comma-separated list).  When --symbols-file is "
                         "set, the LAST positional is taken as OUT and the "
                         "rest are ignored; otherwise the LAST positional "
                         "is OUT and the others are SYMBOL(s).")
    ap.add_argument("--symbols-file", type=Path, default=None,
                    help="File with one symbol per line (blank/# lines "
                         "ignored).  When provided, --out must also be set.")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output single-kernel .sass (required with "
                         "--symbols-file; otherwise inferred from the last "
                         "positional argument).")
    args = ap.parse_args()

    # Resolve out path + symbol list with a back-compat-friendly argv split.
    if args.symbols_file is not None:
        if args.out is None:
            ap.error("--symbols-file requires --out OUT.sass")
        out_path: Path = args.out
        symbols = _parse_symbols(args)
    else:
        if args.out is not None:
            # Explicit --out: every positional is a symbol token.
            out_path = args.out
            symbols = _parse_symbols(args)
        else:
            # Back-compat: at least 2 positionals, last is OUT.
            if len(args.symbols) < 2:
                ap.error("usage: MEGA.sass SYMBOL [SYMBOL ...] OUT.sass  "
                         "(or use --symbols-file with --out)")
            out_path = Path(args.symbols[-1])
            args.symbols = args.symbols[:-1]
            symbols = _parse_symbols(args)

    if not symbols:
        ap.error("no symbols requested")

    ok = extract(args.mega, symbols, out_path)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
