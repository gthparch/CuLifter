# fa_partial_block — FA3 hdim192_128 QK online-softmax rescale repro attempt

Built to reproduce the flash_fwd_hdim192_128_bf16_paged_sm90 ~2% O error
(max_err 0.0387, LSE 0.054), previously diagnosed as the
online-softmax max/sum rescale across the 3 key-blocks (kBlockN=96, partial last
block of 64). meta: D=192 DV=128 SK=256 packgqa=1 paged=1, scale=0.07216878.

## Microbenches (lift: python main.py -i X_sm90.sass X.cubin -o OUT --lifter x86 -name K)
All x86: clang -O2 the .ll, g++ link with test/common/kernel_wrapper_90.h + host.

| bench | what | result |
|---|---|---|
| fa_partial_block.cu / host_cpu.cpp | per-lane online-softmax, KTILE=96 SK=256 partial-block masked via -1e30 | PASS 3e-9 |
| fa3_faithful.cu / host_faithful.cpp | EXACT FA3 structure: reverse iter (partial first), RescaleOBeforeGemm, pipelined scores_scale, exp2(raw*scale-max*scale), Check_inf mask | PASS 5.6e-9 |
| wgmma_n96_reduce.cu / host_n96.cpp | real HGMMA.64x96x16 + FA3 reduce_max/exp2/reduce_sum + cross-quad SHFL.BFLY over 96 cols (K_INTER/SW32 fill) | PASS 3e-5 |
| wgmma_n96_mask.cu / host_n96mask.cpp | n96 + Seqlenk partial-block mask (cols>=64 -> -inf via C-frag col idx) + reduce | PASS 1.9e-5 |
| tma_sw128_k.cu (-DKK=64) / host_tma_sw128.cpp | TMA-filled SW128 (swizzle=3) single-atom HGMMA, raw D dump | PASS 0 (exact) |
| tma_sw128_k.cu (-DKK=128/192) | TMA-filled SW128 multi-atom | FAIL — but see NOTE |
| wgmma_k{64,128,192}_sw128.cu | cute-STS-filled SW128 | FAIL — STS-fill artifact (see NOTE) |

## NOTE — the multi-atom failures are MICROBENCH ARTIFACTS, not the real bug
- cute-STS fill: the lifted STS write swizzle doesn't match the WGMMA _sw128_phys_off
  read for SW128 (cute layout == read formula was probed 0-mismatch, but the lifted
  STS path scrambles). The real kernel fills via TMA/cp.async, not STS.
- cute toy multi-atom WGMMA reuses ONE gdesc reg (UR8) for 10 MMAs; the REAL hdim192
  QK uses 12 DISTINCT gdesc regs (UR16,UR56,UR52,UR48,...), each an independently
  computed per-K-block descriptor. So my toy's K-block stepping != the real kernel's.

## PRUNED (post-campaign tree cleanup, 2026-07)
Only the PASSING (load-bearing) benches above are kept in-tree. The FAIL-artifact
probe variants (`wgmma_k{64,128,192}_sw128.*` + `host_k*.cpp`, the `*_rawd` probes,
`wgmma_n96k192.cu`, and the KK=128/192 `tma_sw128_k*` builds) were deleted —
their conclusions are recorded in the NOTE above; `tma_sw128_k.cu` still builds
any KK via `-DKK=`.

## CONCLUSION
Every component FAITHFUL to the real kernel PASSES. The hdim192 ~2% bug was NOT
reproduced in isolation; the prior "online-softmax rescale across 3 key-blocks"
hypothesis is DISPROVEN (fa3_faithful reproduces that exact algorithm and lifts
byte-correctly). The residual must be in something only the full kernel exercises
(per-HGMMA distinct-descriptor K=192 chain under packgqa/paged cp.async gather, OR
bf16 precision in the running state).
