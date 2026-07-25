#!/usr/bin/env bash
# CPU-only MICRO'26 sm_90 reproduction entry point.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec python "$ROOT/reproduce/run_cpu_sm90.py" "$@"
