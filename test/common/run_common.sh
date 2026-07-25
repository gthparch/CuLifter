#!/bin/bash
# run_common.sh — shared "lift every kernel" logic for the micro-suites.
#
# A per-suite run.sh sources this then calls:  run_suite <suite_dir>
# It lifts every kernel under <suite_dir>/inputs/<K>/ to <K>.ll (x86 backend).
# SASS is derived on demand (cuobjdump) when only a binary is stored.
#
# Env: SLIFTER_ROOT (default: auto-derived repo root), LIFTER (default x86).
set -u
: "${SLIFTER_ROOT:=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
: "${LIFTER:=x86}"

# Directory of this shared lib (works whether sourced directly or via verify_common).
_RC_LIB="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Resolve a cuobjdump that can actually disassemble these (>=sm_90, CUDA-13.x)
# cubins. The stock cuobjdump on many boxes is too old and aborts with
# "Support for 'sm_4' has been removed"; a 13.x disassembler reads them fine.
# Preference order:
#   1. $CUOBJDUMP if the caller set an executable one
#   2. the bundled lib/tools/cuobjdump (+ its paired nvdisasm) shipped here
#   3. plain `cuobjdump` on PATH (last resort — may be too old)
# Side effect: exports NVDISASM_PATH when a bundled nvdisasm is present, since
# cuobjdump shells out to nvdisasm for --dump-sass.
_resolve_cuobjdump() {
    if [ -n "${CUOBJDUMP:-}" ] && [ -x "${CUOBJDUMP}" ]; then
        echo "$CUOBJDUMP"; return 0
    fi
    if [ -x "$_RC_LIB/tools/cuobjdump" ]; then
        [ -x "$_RC_LIB/tools/nvdisasm" ] && export NVDISASM_PATH="$_RC_LIB/tools"
        echo "$_RC_LIB/tools/cuobjdump"; return 0
    fi
    command -v cuobjdump 2>/dev/null || echo cuobjdump
}
CUOBJDUMP="$(_resolve_cuobjdump)"
: "${NVCC_ARCH:=sm_90}"

# Dump SASS from a cubin robustly. The prebuilt library/synth cubins are CUDA-13.x
# and need the bundled 13.x cuobjdump; cubins freshly built by a local (older) nvcc
# need that toolkit's stock cuobjdump. Try the resolved one, then fall back to the
# other candidate if the output is empty. Writes to $2; returns 0 on non-empty.
_dump_sass() {
    local cubin="$1" out="$2"
    "$CUOBJDUMP" --dump-sass "$cubin" > "$out" 2>/dev/null
    if [ ! -s "$out" ]; then
        local alt; alt="$(command -v cuobjdump 2>/dev/null)"
        [ -n "$alt" ] && [ "$alt" != "$CUOBJDUMP" ] && \
            "$alt" --dump-sass "$cubin" > "$out" 2>/dev/null
    fi
    [ -s "$out" ]
}

# Locate the SASS text + (optional) ELF/cubin for one kernel input dir.
# Echoes:  "<sass_path>|<elf_path_or_empty>"
_find_inputs() {
    local d="$1" k="$2"
    local sass="" elf=""
    # Preferred stored SASS, in priority order
    for cand in "$d/kernel.sass" "$d/${k}_sm90.sass" "$d/${k}_sm100.sass" \
                "$d/${k}_sm75.sass" "$d/${k}.sass"; do
        [ -f "$cand" ] && { sass="$cand"; break; }
    done
    for cand in "$d/kernel.elf" "$d/${k}_sm90.elf" "$d/${k}_sm100.elf" \
                "$d/${k}_sm75.elf" "$d/${k}.elf" "$d/kernel.cubin" "$d/${k}.cubin"; do
        [ -f "$cand" ] && { elf="$cand"; break; }
    done
    # Locate a cubin (independently of what the elf-preference loop picked, since a
    # .cubin can sit next to an .elf). If none and only a .cu source exists, build
    # one with nvcc (the ml suite ships source, not binaries).
    local cubin=""
    for cand in "$d/kernel.cubin" "$d/${k}.cubin" "$d/${k}_sm90.cubin" \
                "$d/${k}_sm100.cubin" "$d/${k}_sm75.cubin"; do
        [ -f "$cand" ] && { cubin="$cand"; break; }
    done
    [ -z "$cubin" ] && [[ "$elf" == *.cubin ]] && cubin="$elf"
    if [ -z "$sass" ] && [ -z "$cubin" ] && command -v nvcc >/dev/null 2>&1; then
        local cu=""
        for cand in "$d/${k}.cu" "$d/kernel.cu"; do [ -f "$cand" ] && { cu="$cand"; break; }; done
        [ -z "$cu" ] && cu="$(ls "$d"/*.cu 2>/dev/null | head -1)"
        if [ -n "$cu" ]; then
            cubin="$d/_built.cubin"
            nvcc -arch="$NVCC_ARCH" -cubin "$cu" -o "$cubin" 2>/dev/null || cubin=""
            [ -n "$cubin" ] && elf="$cubin"
        fi
    fi
    # No stored SASS -> derive it from the cubin.
    if [ -z "$sass" ] && [ -n "$cubin" ]; then
        sass="$d/_derived.sass"
        _dump_sass "$cubin" "$sass" || sass=""
    fi
    echo "$sass|$elf"
}

# Sampling knobs honored by run_suite/verify_suite:
#   KERNELS="k1 k2"  -> only these kernel dirs
#   LIMIT=N          -> only the first N kernel dirs (after any KERNELS filter)
# _want_kernel <k> : is kernel k selected?
_want_kernel() {
    local k="$1"
    [ -n "${KERNELS:-}" ] || return 0
    local w
    for w in $KERNELS; do [ "$w" = "$k" ] && return 0; done
    return 1
}

run_suite() {
    local suite_dir="$1"
    local n=0 ok=0
    for d in "$suite_dir"/inputs/*/; do
        [ -d "$d" ] || continue
        local k; k=$(basename "$d")
        _want_kernel "$k" || continue
        [ -n "${LIMIT:-}" ] && [ "$n" -ge "$LIMIT" ] && break
        n=$((n+1))
        local pair sass elf
        pair=$(_find_inputs "$d" "$k"); sass="${pair%%|*}"; elf="${pair##*|}"
        if [ -z "$sass" ]; then echo "[skip] $k (no SASS/cubin)"; continue; fi
        local kf=""
        [ -f "$d/metadata.json" ] && kf=$(grep -oE '"kernel_func"[^,]*' "$d/metadata.json" 2>/dev/null | head -1 | sed 's/.*: *"//;s/".*//')
        local out="$suite_dir/inputs/$k/$k"
        local args=(-i "$sass")
        [ -n "$elf" ] && [[ "$elf" != *.cubin ]] && args+=("$elf")
        [ -n "$elf" ] && [[ "$elf" == *.cubin ]] && args+=("$elf")
        [ -n "$kf" ] && set -- -name "$kf" || set --
        if python "$SLIFTER_ROOT/main.py" "${args[@]}" -o "$out" --lifter "$LIFTER" "$@" \
                >"$out.lift.log" 2>&1; then
            echo "[lift OK] $k"; ok=$((ok+1))
        else
            echo "[LIFT-FAIL] $k -> $(tail -1 "$out.lift.log")"
        fi
    done
    echo "----- $(basename "$suite_dir"): $ok/$n lifted -----"
}
