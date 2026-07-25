# nv_libraries — CPU execution-correctness (verified kernels)

Each kernel: lifted by SLifter (SM90 SASS → x86 IR), run on CPU, checked against an
**independent host reference** (decoded from the SASS semantics / known algorithm,
NOT copied from the lifted IR) on discriminating inputs. FP-derived outputs use
`nvlib_compare.h` tolerance (±1 uint8 / atol+rtol float). Every PASS confirmed by a
from-scratch `verify.sh` re-run (final authoritative sweep).

## 98 distinct kernels EXACT execution-correct  (+3 sound-only = 101 pass verify.sh)

### nvjpeg color / chroma conversion — 82 (via parameterized generators)
| family | n | generator | what differs across variants |
|---|---|---|---|
| `ycbcr_to_format` | 32 | `gen_ycbcr.py <subsamp> <fmt>` | subsampling {444,422,420,440,411,410,GRAY,410V} × format {RGB,BGR,RGBI,BGRI}; FFMA color conv (fmaf), planar + interleaved output |
| `format_to_ycbcr` (planar in) | 14 | `gen_format_to_ycbcr.py <infmt> <subsamp>` | input {RGB,BGR} × subsampling {0..6}; RGB→Y/Cb/Cr (JFIF), chroma averaged per block |
| `subsample_chroma` | 36 | (agent) | src×dst chroma subsampling {0..5}²; box-average downsample / replicate upsample of Cb,Cr |

(interleaved-input `format_to_ycbcr` (14) deferred — an input-addressing mismatch
under the interleaved layout; not a lift bug.)

### npp distance-transform (PBA) + misc — 9
kernelColor (shared-mem transpose+barrier), CreateForwardPointers, PropagateInterband,
MergeBands (64-bit dominate), FloodDown, FloodUp, ProximatePoints, UpdateVertical
(shared-mem transpose+barrier), DoubleToSingleList.

### npp_uf — 2 exact  (labelMarkersUFUpdate, UF8WayLocalMergeBlock)
### npp_compress — 5  (Pass1, Pass3, CountBlockLocalActiveIndexesUF, SumLocalBlockCountsUF, OffsetLocalBlockCountsByBlockSumBaseUF)

### + 3 sound-but-not-exact (npp_uf 8-way GlobalMerge / GlobalMergePartial / BoundaryAnalysis)
Single stages of NPP's *iterative* union-find pipeline — one standalone pass from
identity init doesn't converge to a full labelling, so their drivers verify
SOUNDNESS (no cross-component merge) + determinism, not exact labels.
(GlobalMergePartial is additionally racy under concurrent CPU threads.)

## 2 lifter bugs found AND FIXED (lift/lifter.py)
- `lift_i2f` dropped operand byte-select `.B1/.B2/.B3` → fixed (shift lane to low bits).
- `lift_fmul` dropped `.D2/.D4/.D8` result divisor → fixed (×0.5/0.25/0.125).
Surfaced by `format_to_ycbcr` (was 128/192 wrong → 0). Both fixed in `lift/lifter.py`.

## Method / tooling
Parameterized generators (`gen_ycbcr.py`, `gen_format_to_ycbcr.py`) cover the
large template families; dedicated drivers cover `subsample_chroma`, NPP PBA,
union-find, and `npp_compress`. Each was audited by re-running `verify.sh`.
`RECIPE.md` (method), `verify.sh` (NVLIB_LIFT_CACHE=1 reuses kernel.ll),
`nvlib_compare.h` (tolerance), `autogen_harness.py` (run-safety auto-harness).

## Still BLOCKED (not lift bugs)
batched_ycbcr2rgb (144-byte pointer-to-struct Param), nvjpeg2k tier1/inv_quantize
(descriptor structs), nvtiff LZW (needs valid bitstream), curand seed (RNG
reproducibility), and IDCT by-value. Each needs a hand-built valid Params/input.
