#!/usr/bin/env bash
# Canonical artifact entry point.
#
#   bash reproduce/run_all.sh                    # E1 then E2
#   bash reproduce/run_all.sh --jobs 4           # change E1 parallelism
#   bash reproduce/run_all.sh --only e1           # lifting coverage only
#   bash reproduce/run_all.sh --only e2           # figure data only
#
# E1 lifts the complete benchmark manifest. E2 regenerates all data-driven
# paper figures.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH="${SLIFTER_BENCHMARK_DIR:-$ROOT/benchmark}"
ONLY=all
LIFT_ARGS=()
FIG_ARGS=()

die() {
    echo "ERROR: $*" >&2
    exit 2
}

while [ $# -gt 0 ]; do
    case "$1" in
        --jobs)
            [ $# -ge 2 ] || die "--jobs requires a value"
            LIFT_ARGS+=(--jobs "$2")
            shift 2
            ;;
        --suite)
            [ $# -ge 2 ] || die "--suite requires a value"
            LIFT_ARGS+=(--suite "$2")
            shift 2
            ;;
        --only)
            [ $# -ge 2 ] || die "--only requires e1, e2, or all"
            ONLY="$2"
            shift 2
            ;;
        -h|--help)
            sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            die "unknown argument: $1 (see --help)"
            ;;
    esac
done

case "$ONLY" in
    e1|e2|all) ;;
    *) die "--only requires e1, e2, or all" ;;
esac

[ -d "$BENCH" ] || {
    echo "ERROR: benchmark bundle not found at $BENCH" >&2
    echo "       run: bash reproduce/setup_data.sh" >&2
    exit 1
}

STAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR="$ROOT/reproduce/results/run_all_${STAMP}_$$"
mkdir -p "$LOGDIR"

echo "======================================================================"
echo "  CuLifter artifact evaluation      logs -> $LOGDIR"
echo "  bundle: $BENCH"
echo "======================================================================"

t0=$(date +%s)
E1_RC=skipped
E2_RC=skipped

if [ "$ONLY" = e1 ] || [ "$ONLY" = all ]; then
    echo
    echo "### E1 — full benchmark lifting coverage"
    bash "$ROOT/reproduce/run_cpu_sm90.sh" "${LIFT_ARGS[@]}" 2>&1 \
        | tee "$LOGDIR/e1_lift.log"
    E1_RC=${PIPESTATUS[0]}
fi

if [ "$ONLY" = e2 ] || [ "$ONLY" = all ]; then
    echo
    echo "### E2 — paper figure data"
    bash "$ROOT/reproduce/figures/reproduce_figures.sh" \
        "${FIG_ARGS[@]}" 2>&1 | tee "$LOGDIR/e2_figures.log"
    E2_RC=${PIPESTATUS[0]}
fi

elapsed=$(( $(date +%s) - t0 ))
echo
echo "======================================================================"
echo "  SUMMARY                          elapsed ${elapsed}s"
echo "======================================================================"

if [ "$E1_RC" != skipped ]; then
    [ "$E1_RC" -eq 0 ] \
        && echo "E1: completed (rc=0)" \
        || echo "E1: FAIL (rc=$E1_RC; see $LOGDIR/e1_lift.log)"
fi
if [ "$E2_RC" != skipped ]; then
    [ "$E2_RC" -eq 0 ] \
        && echo "E2: completed (rc=0)" \
        || echo "E2: FAIL (rc=$E2_RC; see $LOGDIR/e2_figures.log)"
fi
echo "logs: $LOGDIR"

[ "$E1_RC" = skipped ] || [ "$E1_RC" -eq 0 ] || exit "$E1_RC"
[ "$E2_RC" = skipped ] || [ "$E2_RC" -eq 0 ] || exit "$E2_RC"
