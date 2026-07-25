#!/usr/bin/env bash
# Install and validate the separately distributed benchmark bundle.
#
#   bash reproduce/setup_data.sh                  # download the pinned archive
#   bash reproduce/setup_data.sh /path/to/culifter_benchmarks.zip
#   SLIFTER_BUNDLE_URL=https://... bash reproduce/setup_data.sh
#   bash reproduce/setup_data.sh --force         # replace an existing benchmark/
#   bash reproduce/setup_data.sh --check         # verify an existing benchmark/ only
#
# The default Dropbox URL is pinned below. The checked-in checksum and
# reproduction manifest identify the submitted archive and all evaluated files.
#
#   export SLIFTER_BUNDLE_URL='https://.../culifter_benchmarks.zip'
#
# Google Drive links work too — either the share URL or the bare file id:
#
#   export SLIFTER_BUNDLE_URL='https://drive.google.com/file/d/<ID>/view?usp=sharing'
#   export SLIFTER_BUNDLE_GDRIVE_ID='<ID>'
#
# Drive downloads need `gdown` (pip install gdown); plain HTTPS needs curl or wget.
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
HOME_REAL="$(realpath -m "$HOME")"
BENCH_INPUT="${SLIFTER_BENCHMARK_DIR:-$ROOT/benchmark}"
[ ! -L "$BENCH_INPUT" ] || {
    echo "ERROR: benchmark target must not be a symbolic link: $BENCH_INPUT" >&2
    exit 1
}
BENCH="$(realpath -m "$BENCH_INPUT")"
case "$BENCH" in
    /|"$HOME_REAL"|"$ROOT")
        echo "ERROR: unsafe benchmark target: $BENCH" >&2
        exit 1
        ;;
esac
[ "$(basename "$BENCH")" = benchmark ] || {
    echo "ERROR: benchmark target must end in /benchmark: $BENCH" >&2
    exit 1
}
CHECKSUM_FILE="$ROOT/reproduce/culifter_benchmarks.sha256"
DEFAULT_BUNDLE_URL='https://www.dropbox.com/scl/fi/jgxafm9jxmxcm7jcrh7x3/culifter_benchmarks.zip?rlkey=mldtd27otou1zzx0lzghmws56&st=hxa4f55g&dl=1'
BUNDLE_URL="${SLIFTER_BUNDLE_URL:-$DEFAULT_BUNDLE_URL}"
GDRIVE_ID="${SLIFTER_BUNDLE_GDRIVE_ID:-}"
ZIP=""; FORCE=0; CHECK_ONLY=0; KEEP_ZIP=1

SUITES="hecbench cublas cutlass flash_attention cudnn_cnn cudnn_ops cudnn_adv cuda_sdk"

die()  { echo "ERROR: $*" >&2; exit 1; }
info() { echo "[setup] $*"; }

while [ $# -gt 0 ]; do
    case "$1" in
        --force) FORCE=1; shift;;
        --check) CHECK_ONLY=1; shift;;
        --drop-zip) KEEP_ZIP=0; shift;;
        -h|--help) awk 'NR>1 && /^#/ {sub(/^# ?/,""); print; next} NR>1 {exit}' "$0"; exit 0;;
        -*) die "unknown arg: $1 (see --help)";;
        *)
            [ -z "$ZIP" ] || die "pass only one benchmark archive"
            ZIP="$1"
            shift
            ;;
    esac
done

# --- verification -------------------------------------------------------------
verify() {
    local missing=0 s
    for s in $SUITES; do
        if [ -d "$BENCH/$s" ]; then
            printf '  %-18s %s\n' "$s" "$(du -sh "$BENCH/$s" 2>/dev/null | cut -f1)"
        else
            printf '  %-18s MISSING\n' "$s"; missing=1
        fi
    done
    [ "$missing" -eq 0 ] || return 1
    SLIFTER_BENCHMARK_DIR="$BENCH" \
        bash "$ROOT/reproduce/run_cpu_sm90.sh" --check
}

if [ "$CHECK_ONLY" = 1 ]; then
    info "checking $BENCH"
    if verify; then
        info "bundle OK"
        exit 0
    fi
    die "bundle incomplete; install the exact submission archive"
fi

if [ -e "$BENCH" ] && [ "$FORCE" = 0 ]; then
    info "benchmark/ already present at $BENCH (use --force to replace it)"
    [ -d "$BENCH/cublas" ] \
        && verify \
        && { info "bundle OK — nothing to do"; exit 0; }
    die "an incomplete bundle already exists; inspect it, then rerun with --force"
fi

# --- locate the archive -------------------------------------------------------
CACHE="${SLIFTER_BUNDLE_CACHE:-$ROOT/.bundle_cache}"
if [ -z "$ZIP" ]; then
    mkdir -p "$CACHE"
    ZIP="$CACHE/culifter_benchmarks.zip"
    if [ -s "$ZIP" ]; then
        info "reusing cached archive $ZIP"
    else
        # a Drive share URL carries the file id in the path
        if [ -z "$GDRIVE_ID" ] && [[ "$BUNDLE_URL" == *drive.google.com* ]]; then
            GDRIVE_ID=$(printf '%s' "$BUNDLE_URL" | sed -nE 's#.*/d/([^/]+).*#\1#p')
            [ -n "$GDRIVE_ID" ] || GDRIVE_ID=$(printf '%s' "$BUNDLE_URL" | sed -nE 's#.*[?&]id=([^&]+).*#\1#p')
        fi

        info "downloading the benchmark bundle (several GB — this takes a while)"
        if [ -n "$GDRIVE_ID" ]; then
            command -v gdown >/dev/null 2>&1 || die \
"Google Drive download needs gdown:  pip install gdown
       (or download the zip by hand and pass its path to this script)"
            gdown --id "$GDRIVE_ID" -O "$ZIP" || die "download failed"
        elif command -v curl >/dev/null 2>&1; then
            curl -fL --retry 3 -C - -o "$ZIP" "$BUNDLE_URL" || die "download failed"
        elif command -v wget >/dev/null 2>&1; then
            wget -c -O "$ZIP" "$BUNDLE_URL" || die "download failed"
        else
            die "need curl or wget to download $BUNDLE_URL"
        fi
    fi
fi
[ -s "$ZIP" ] || die "archive not found: $ZIP"

[ -f "$CHECKSUM_FILE" ] || die "missing checked-in archive checksum: $CHECKSUM_FILE"
read -r expected_sha expected_name <"$CHECKSUM_FILE"
[[ "$expected_sha" =~ ^[[:xdigit:]]{64}$ ]] \
    || die "invalid checksum in $CHECKSUM_FILE"
[ "$expected_name" = "culifter_benchmarks.zip" ] \
    || die "unexpected archive name in $CHECKSUM_FILE: $expected_name"
info "verifying archive sha256"
have=$(sha256sum "$ZIP" | cut -d' ' -f1)
[ "$have" = "$expected_sha" ] \
    || die "checksum mismatch: got $have, expected $expected_sha"

# --- extract ------------------------------------------------------------------
bench_parent=$(dirname "$BENCH")
mkdir -p "$bench_parent"
avail=$(df -Pk "$bench_parent" | awk 'NR==2{print int($4/1048576)}')
[ "$avail" -ge 20 ] || info \
    "WARNING: only ${avail} GB free; the full install-and-run workflow needs about 20 GB"

command -v unzip >/dev/null 2>&1 || die "unzip not found"
stage=$(mktemp -d "$bench_parent/.culifter-install.XXXXXX") \
    || die "could not create an installation staging directory"
cleanup_stage() {
    if [ -n "${stage:-}" ] && [ -d "$stage" ]; then
        rm -rf -- "$stage"
    fi
}
trap cleanup_stage EXIT

info "extracting to a clean staging directory (about 4.7 GB)"
unzip -q "$ZIP" -d "$stage" || die "extraction failed"
stage_bench="$stage/benchmark"
[ -d "$stage_bench" ] || die \
"archive must contain one top-level benchmark/ directory.
       Check its layout with: unzip -l $ZIP | head"

info "validating staged corpus"
SLIFTER_BENCHMARK_DIR="$stage_bench" \
    bash "$ROOT/reproduce/run_cpu_sm90.sh" --check \
    || die "staged bundle failed manifest validation"

backup=
if [ -e "$BENCH" ]; then
    [ "$FORCE" = 1 ] || die "$BENCH already exists; use --force to replace it"
    backup="$bench_parent/.culifter-backup.$(date +%Y%m%d_%H%M%S).$$"
    mv -- "$BENCH" "$backup" || die "could not stage the existing benchmark directory"
fi
if ! mv -- "$stage_bench" "$BENCH"; then
    [ -z "$backup" ] || mv -- "$backup" "$BENCH"
    die "could not install the validated benchmark directory"
fi
if [ -n "$backup" ]; then
    rm -rf -- "$backup"
fi
info "installed validated corpus at $BENCH"

if [ "$KEEP_ZIP" = 0 ]; then
    cache_real=$(realpath -m "$CACHE")
    zip_real=$(realpath -m "$ZIP")
    case "$cache_real" in
        /|"$HOME_REAL"|"$ROOT")
            die "unsafe bundle cache for --drop-zip: $cache_real"
            ;;
    esac
    case "$zip_real" in
        "$cache_real"/*)
            rm -f -- "$zip_real"
            info "removed cached archive"
            ;;
        *)
            info "archive is outside the bundle cache; keeping $ZIP"
            ;;
    esac
else
    info "archive kept at $ZIP (delete it to reclaim space)"
fi

echo
info "done — next: bash reproduce/run_all.sh"
