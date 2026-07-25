#!/bin/bash
# run.sh — lift-only: disassemble each binary and lift to NVVM IR.
# Usage: SLIFTER_ROOT=<lifter checkout> [LIFTER=nvvm] bash run.sh
# The CUDA-13.2 cublas/cudnn cubins need a 13.x disassembler; the bundled
# lib/tools/cuobjdump (+nvdisasm) supplies it (an old stock cuobjdump aborts with
# "Support for 'sm_4' has been removed"). The .elf text dumps lift directly.
set -u
: "${SLIFTER_ROOT:=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
: "${LIFTER:=nvvm}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Reuse the shared cuobjdump resolver (prefers bundled lib/tools/cuobjdump).
source "$HERE/../common/run_common.sh"
n=0; ok=0
shopt -s nullglob
for bin in "$HERE"/cubins/*.cubin "$HERE"/elf/*.elf; do
    [ -f "$bin" ] || continue
    n=$((n+1)); base=$(basename "$bin"); sass="$HERE/${base}.sass"
    if [[ "$bin" == *.cubin ]]; then
        "$CUOBJDUMP" --dump-sass "$bin" > "$sass" 2>"$sass.err" || {
            echo "[DISASM-FAIL] $base: $(tail -1 "$sass.err")"; continue; }
        [ -s "$sass" ] || { echo "[DISASM-EMPTY] $base"; continue; }
        inp=("$sass" "$bin")
    else
        inp=("$bin")   # .elf text dumps lift as a single input
    fi
    if python "$SLIFTER_ROOT/main.py" -i "${inp[@]}" -o "$HERE/${base}" --lifter "$LIFTER" \
            >"$HERE/${base}.lift.log" 2>&1; then
        echo "[lift OK] $base"; ok=$((ok+1))
    else
        echo "[LIFT-FAIL] $base: $(tail -1 "$HERE/${base}.lift.log")"
    fi
done
echo "----- $(basename "$HERE"): $ok/$n lifted -----"
