#!/usr/bin/env python3
"""Split a multi-kernel cuobjdump SASS file into per-kernel files.

Usage:
    python scripts/utils/split_sass.py INPUT.sass OUTPUT_DIR [--isa sm_90]

Each `Function : <name>` block becomes its own file at OUTPUT_DIR/<safe_name>.sass
with the ISA header prepended so SLifter can parse it standalone.
"""

import argparse
import re
import sys
from pathlib import Path


def sanitize_name(name, max_len=200):
    """Convert kernel name to a safe filename."""
    safe = re.sub(r'[^\w\-.]', '_', name)
    if len(safe) > max_len:
        import hashlib
        h = hashlib.md5(name.encode()).hexdigest()[:8]
        safe = safe[:max_len - 9] + '_' + h
    return safe


def split_sass(input_path, output_dir, isa="sm_90"):
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    text = input_path.read_text()
    lines = text.split('\n')

    header = f"\n\tcode for {isa}\n"

    kernels = []
    current_name = None
    current_lines = []

    for line in lines:
        m = re.match(r'\s+Function\s*:\s*(.+)', line)
        if m:
            # Save previous kernel
            if current_name is not None:
                kernels.append((current_name, current_lines))
            current_name = m.group(1).strip()
            current_lines = [line]
        elif current_name is not None:
            current_lines.append(line)

    # Save last kernel
    if current_name is not None:
        kernels.append((current_name, current_lines))

    # Write per-kernel files
    manifest = []
    for name, klines in kernels:
        safe = sanitize_name(name)
        out_path = output_dir / f"{safe}.sass"
        with open(out_path, 'w') as f:
            f.write(header)
            for kl in klines:
                f.write(kl + '\n')
        manifest.append((safe, name))

    # Write manifest
    manifest_path = output_dir / "_manifest.txt"
    with open(manifest_path, 'w') as f:
        for safe, orig in manifest:
            f.write(f"{safe}\t{orig}\n")

    print(f"Split {len(kernels)} kernels from {input_path.name} into {output_dir}")
    return len(kernels)


def main():
    parser = argparse.ArgumentParser(description="Split multi-kernel SASS into per-kernel files")
    parser.add_argument("input", help="Input SASS file (cuobjdump format)")
    parser.add_argument("output_dir", help="Output directory for per-kernel files")
    parser.add_argument("--isa", default="sm_90", help="ISA string (default: sm_90)")
    args = parser.parse_args()

    n = split_sass(args.input, args.output_dir, args.isa)
    print(f"Done. {n} kernels written.")


if __name__ == "__main__":
    main()
