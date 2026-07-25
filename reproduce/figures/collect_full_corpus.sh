#!/usr/bin/env bash
# Full-corpus characterization: run the single-pass figure collector over
# every unit of reproduce/sm90_manifest.json.
#
#   bash reproduce/figures/collect_full_corpus.sh [suite ...]
#
# Per-suite wall time and peak RSS are recorded in _data/logs/stage_times.tsv.
# Heavy library suites run at reduced parallelism to bound peak memory.
set -eu
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${SLIFTER_PY:-$ROOT/.conda_env/bin/python}"
[ -x "$PY" ] || { echo "ERROR: python not found at $PY (set SLIFTER_PY)" >&2; exit 1; }

export SLIFTER_FIG_DISCOVERY=manifest
export SLIFTER_EVAL_TIMEOUT="${SLIFTER_EVAL_TIMEOUT:-3600}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
DATA="${SLIFTER_FIG_DATA:-$ROOT/reproduce/figures/_data}"
LOGS="$DATA/logs"
mkdir -p "$LOGS"
TSV="$LOGS/stage_times.tsv"
[ -f "$TSV" ] || printf 'stage\twall_s\tmax_rss_kib\tdata_kib_after\tstatus\n' > "$TSV"

# suite -> worker count (heavy vendor modules need memory headroom)
declare -A JOBS=(
    [hecbench]=16 [cuobjdump]=8 [cutlass]=12 [cublas]=5
    [cudnn_cnn]=4 [cudnn_ops]=4 [cudnn_adv]=4 [flash_attn]=6
)
ORDER=(cuobjdump hecbench cutlass cudnn_cnn cudnn_ops cudnn_adv flash_attn cublas)
want=("$@")
[ ${#want[@]} -eq 0 ] && want=("${ORDER[@]}")

echo "corpus preflight (manifest vs bundle):"
"$PY" "$ROOT/reproduce/figures/engines/manifest_bundle.py"

for s in "${want[@]}"; do
    j="${JOBS[$s]:-8}"
    echo "============================================================"
    echo "  suite $s  (workers: $j)  $(date '+%F %T')"
    echo "============================================================"
    t0=$(date +%s)
    status=ok
    /usr/bin/time -v nice -n 19 "$PY" -m reproduce.figures.engines.collect_figure_stats \
        --suite "$s" -j "$j" > "$LOGS/$s.log" 2> "$LOGS/$s.time" || status=fail
    wall=$(( $(date +%s) - t0 ))
    rss=$(awk -F': ' '/Maximum resident set size/ {print $2}' "$LOGS/$s.time" | tail -1)
    datak=$(du -sk "$DATA" | cut -f1)
    printf '%s\t%s\t%s\t%s\t%s\n' "collect:$s" "$wall" "${rss:-?}" "$datak" "$status" >> "$TSV"
    tail -2 "$LOGS/$s.log" || true
    [ "$status" = ok ] || echo "WARNING: suite $s exited nonzero (see $LOGS/$s.log)"
done

echo "done: $(date '+%F %T')"
column -t "$TSV"
if grep -q "	fail$" "$TSV"; then
    echo "ERROR: one or more suites failed (see _data/logs)" >&2
    exit 1
fi
