#!/usr/bin/env bash
# One-command regeneration of paper Figures 4-10 over the benchmark
# corpus described by reproduce/sm90_manifest.json.
#
#   bash reproduce/figures/reproduce_figures.sh            # everything
#   bash reproduce/figures/reproduce_figures.sh collect    # one stage
#
# Stages (each timed into _data/logs/stage_times.tsv):
#   collect   figs 4,5,6,7a,7b,10 characterization (single lift pass, SM90)
#   ablation  fig 8a HeCBench B0-B4 (x86 execution, needs clang/llc/g++)
#   nvlib     fig 8b NPP/nvJPEG B0-B4 (x86 execution)
#   degrade   fig 9 seed-removal sweep
#   export    plot-ready CSVs -> reproduce/figures/output/
#
set -eu
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${SLIFTER_PY:-$ROOT/.conda_env/bin/python}"
DATA="${SLIFTER_FIG_DATA:-$ROOT/reproduce/figures/_data}"
LOGS="$DATA/logs"; mkdir -p "$LOGS"
TSV="$LOGS/stage_times.tsv"
[ -f "$TSV" ] || printf 'stage\twall_s\tmax_rss_kib\tdata_kib_after\tstatus\n' > "$TSV"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export SLIFTER_FIG_DISCOVERY=manifest

record() {  # record <stage> <wall> <timefile> <status>
    local rss
    rss=$(awk -F': ' '/Maximum resident set size/ {print $2}' "$3" 2>/dev/null | tail -1)
    printf '%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "${rss:-?}" "$(du -sk "$DATA" | cut -f1)" "$4" >> "$TSV"
}

timed() {  # timed <stage> <cmd...>
    local stage=$1; shift
    local t0 status=ok
    t0=$(date +%s)
    /usr/bin/time -v nice -n 19 "$@" > "$LOGS/$stage.log" 2> "$LOGS/$stage.time" || status=fail
    record "$stage" "$(( $(date +%s) - t0 ))" "$LOGS/$stage.time" "$status"
    [ "$status" = ok ] || { echo "STAGE $stage FAILED — see $LOGS/$stage.log" >&2; return 1; }
}

stage_collect()  { bash "$ROOT/reproduce/figures/collect_full_corpus.sh"; }
stage_ablation() { timed ablation "$PY" "$ROOT/reproduce/figures/engines/run_ablation.py" -j 6 \
                       --output "$DATA/ablation" --work-dir "$ROOT/temp/ablation"; }
stage_nvlib()    { timed nvlib "$PY" "$ROOT/reproduce/figures/engines/run_nvlib_ablation.py" -j 4 \
                       --output "$DATA/ablation_nvlib" --work-dir "$ROOT/temp/nvlib_ablation"; }
stage_degrade()  { timed degrade "$PY" "$ROOT/reproduce/figures/engines/run_type_degradation.py" \
                       -j 8 --output "$DATA/degradation"; }
stage_export()   { timed export "$PY" "$ROOT/reproduce/figures/engines/export_paper_csvs.py"; }

ALL=(collect ablation nvlib degrade export)
want=("$@")
[ ${#want[@]} -eq 0 ] && want=("${ALL[@]}")
for s in "${want[@]}"; do
    case " ${ALL[*]} " in
        *" $s "*) ;;
        *) echo "unknown stage: $s (available: ${ALL[*]})" >&2; exit 2;;
    esac
    echo "=== stage $s $(date '+%F %T') ==="
    "stage_$s"
done
echo "=== done $(date '+%F %T') ==="
column -t "$TSV"
