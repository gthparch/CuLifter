#!/bin/bash
# verify_common.sh — shared binref verify logic for the micro-suites.
#
# A per-suite verify.sh sources this then calls:  verify_suite <suite_dir> <sm>
# Per kernel: lift -> clang -O2 -target x86_64 -c -> g++ link drivers/<K>/host_cpu.cpp
#   + lib/kernel_wrapper_<sm>.h -> run from inputs/<K>/ -> assert "TEST PASSED".
#
# This is the binref method (host self-checks against inputs/<K>/ref_output.bin).
# Kernels with a driver but no ref_output (cuda_handwritten/ml/sm100/cuobjdump)
# pass on "TEST PASSED" OR "REF VERIFIED" the host prints; lift-only kernels (no
# host_cpu.cpp) are reported LIFT-ONLY.
set -u
: "${SLIFTER_ROOT:=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
LIB="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$LIB/run_common.sh"

verify_suite() {
    local suite_dir="$1" sm="${2:-90}"
    local wrapper="$LIB/kernel_wrapper_${sm}.h"
    local n=0 pass=0 liftonly=0
    for d in "$suite_dir"/inputs/*/; do
        [ -d "$d" ] || continue
        local k; k=$(basename "$d")
        _want_kernel "$k" || continue
        [ -n "${LIMIT:-}" ] && [ "$n" -ge "$LIMIT" ] && break
        n=$((n+1))
        local drv="$suite_dir/drivers/$k/host_cpu.cpp"
        [ -f "$drv" ] || drv="$suite_dir/drivers/$k/host.cpp"
        # 1. lift
        local pair sass elf; pair=$(_find_inputs "$d" "$k")
        sass="${pair%%|*}"; elf="${pair##*|}"
        [ -z "$sass" ] && { echo "[skip] $k (no SASS)"; continue; }
        local kf=""; [ -f "$d/metadata.json" ] && kf=$(grep -oE '"kernel_func"[^,]*' "$d/metadata.json" 2>/dev/null | head -1 | sed 's/.*: *"//;s/".*//')
        local work; work=$(mktemp -d)
        local args=(-i "$sass"); [ -n "$elf" ] && args+=("$elf")
        local nm=(); [ -n "$kf" ] && nm=(-name "$kf")
        if ! python "$SLIFTER_ROOT/main.py" "${args[@]}" -o "$work/k" --lifter x86 "${nm[@]}" \
                >"$work/lift.log" 2>&1; then
            echo "[LIFT-FAIL] $k: $(tail -1 "$work/lift.log")"; rm -rf "$work"; continue
        fi
        if [ ! -f "$drv" ]; then echo "[lift-only] $k (no host driver)"; liftonly=$((liftonly+1)); rm -rf "$work"; continue; fi
        # 2. compile + link
        clang -O2 -fPIC -target x86_64 -c "$work/k.ll" -o "$work/k.o" 2>"$work/clang.log" \
            || { echo "[CLANG-FAIL] $k"; rm -rf "$work"; continue; }
        local inc; inc=$(mktemp -d); cp "$wrapper" "$inc/kernel_wrapper.h"; cp "$LIB"/kernel_wrapper_*.h "$inc/" 2>/dev/null
        # Stage the driver one level under a dir holding the suite-root headers, so
        # a driver's "../common_ml.h"-style include (written for a flat layout)
        # resolves. Copy the whole driver dir (keeps any sibling headers too).
        mkdir -p "$work/stage/$k"
        cp "$suite_dir/drivers/$k/"* "$work/stage/$k/" 2>/dev/null
        cp "$suite_dir"/*.h "$work/stage/" 2>/dev/null
        g++ -std=c++20 -O2 "-D__CUDA_ARCH__=${sm}0" -I"$inc" -I"$LIB" -I"$d" -I"$suite_dir" \
            "$work/stage/$k/$(basename "$drv")" "$work/k.o" -o "$work/run" -lpthread 2>"$work/gpp.log" \
            || { echo "[GPP-FAIL] $k: $(tail -1 "$work/gpp.log")"; rm -rf "$work" "$inc"; continue; }
        rm -rf "$inc"
        # 3. run from the input dir (host reads relative ref_output.bin)
        local out; out=$( cd "$d" && timeout 120 "$work/run" 2>&1 )
        if echo "$out" | grep -qE 'TEST PASSED|REF VERIFIED'; then
            echo "[PASS] $k"; pass=$((pass+1))
        else
            echo "[FAIL] $k: $(echo "$out" | tail -1)"
        fi
        rm -rf "$work"
    done
    echo "----- $(basename "$suite_dir"): $pass passed, $liftonly lift-only, $n total -----"
}
