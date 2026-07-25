#!/bin/bash
# verify.sh — LIFT-ONLY suite: there is no output reference for these closed
# library / template binaries. "Verification" == the lifter emits valid IR for
# every kernel. This wrapper just runs run.sh; success = 0 LIFT-FAIL lines.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "[verify] $(basename "$HERE") uses the LIFT-ONLY method (no exec reference)."
exec bash "$HERE/run.sh"
