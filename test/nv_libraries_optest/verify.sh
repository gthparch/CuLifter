#!/usr/bin/env bash
# verify.sh <test-name> <cubin-basename> <mangled-name>
# Extract one kernel, lift to x86, build its committed host driver, and run it.
set -euo pipefail

if [ "$#" -ne 3 ]; then
    echo "usage: $0 TEST_NAME CUBIN_BASENAME MANGLED_NAME" >&2
    exit 2
fi
TEST_NAME="$1"
CUBIN_NAME="$2"
MANGLED_NAME="$3"
[[ "$TEST_NAME" =~ ^[A-Za-z0-9_.-]+$ ]] || {
    echo "ERROR: TEST_NAME must be one directory name" >&2
    exit 2
}

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT="$(cd "$HERE/../.." && pwd -P)"
BENCHMARK="${SLIFTER_BENCHMARK_DIR:-$ROOT/benchmark}"
CUOBJDUMP="${CUOBJDUMP:-cuobjdump}"
PYTHON="${PYTHON:-python}"
TEST_DIR="$HERE/$TEST_NAME"
CUBIN="$BENCHMARK/nv_libraries/cubins/$CUBIN_NAME.cubin"

[ -f "$CUBIN" ] || {
    echo "ERROR: missing development cubin: $CUBIN" >&2
    exit 2
}
[ -f "$TEST_DIR/host_cpu.cpp" ] || {
    echo "ERROR: missing committed driver: $TEST_DIR/host_cpu.cpp" >&2
    exit 2
}
command -v "$CUOBJDUMP" >/dev/null 2>&1 || {
    echo "ERROR: cuobjdump not found; set CUOBJDUMP" >&2
    exit 2
}

cp "$CUBIN" "$TEST_DIR/kernel.elf"
full_sass="$TEST_DIR/.full.sass"
"$CUOBJDUMP" --dump-sass "$CUBIN" >"$full_sass"
awk -v kernel="$MANGLED_NAME" '
  /code for sm_90/{header=$0}
  /Function :/{keep=($3==kernel); if(keep&&!printed){print "\t" header; printed=1}}
  keep{print}' "$full_sass" >"$TEST_DIR/kernel.sass"
rm -f -- "$full_sass"

function_count=$(grep -c 'Function :' "$TEST_DIR/kernel.sass" || true)
[ "$function_count" -eq 1 ] || {
    echo "ERROR: extracted $function_count functions; expected one for $MANGLED_NAME" >&2
    exit 3
}

if [ "${NVLIB_LIFT_CACHE:-0}" != 1 ] || [ ! -s "$TEST_DIR/kernel.ll" ]; then
    "$PYTHON" "$ROOT/main.py" \
        -i "$TEST_DIR/kernel.sass" "$TEST_DIR/kernel.elf" \
        -o "$TEST_DIR/kernel" --lifter x86 \
        >"$TEST_DIR/.lift.log" 2>&1 || {
            tail -n 5 "$TEST_DIR/.lift.log" >&2
            exit 4
        }
fi

clang -O2 -fPIC -target x86_64 -c "$TEST_DIR/kernel.ll" \
    -o "$TEST_DIR/kernel.o"
include_dir=$(mktemp -d "$HERE/../common/inc.XXXXXX")
cleanup() {
    rm -rf -- "$include_dir"
}
trap cleanup EXIT
cp "$HERE/../common/kernel_wrapper_90.h" "$include_dir/kernel_wrapper.h"
g++ -std=c++20 -O2 -D__CUDA_ARCH__=900 \
    -I"$include_dir" -I"$HERE/../common" -I"$HERE" \
    "$TEST_DIR/host_cpu.cpp" "$TEST_DIR/kernel.o" \
    -o "$TEST_DIR/run" -lpthread 2>"$TEST_DIR/.build.log" || {
        tail -n 8 "$TEST_DIR/.build.log" >&2
        exit 5
    }

output=$("$TEST_DIR/run" 2>&1)
printf '%s\n' "$output" | grep -vE '^ (ptr|u32|u64|raw)|Arg byte_off' || true
grep -q "TEST PASSED" <<<"$output" || {
    echo "FAIL[$TEST_NAME]" >&2
    exit 6
}
echo "PASS[$TEST_NAME]"
