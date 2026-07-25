// Microbench: the 45_dual_gemm grid-strided index-division WIDE-LEAF shape.
//
// coord0 = index / N ; coord1 = index % N  with a RUNTIME UNSIGNED divisor N
// (cutlass uses unsigned/size_t coords). ptxas lowers /N,%N to the
// Granlund-Moller magic-rcp chain whose quotient leaf is `IMAD.HI.U32 q, rcp,
// dividend, RZ`; the corrected quotient (coord0) is then used as a multiplicand
// in a 64-bit ROW-MAJOR address:
//   off = coord0 * (long)stride + coord1   =>  IMAD.WIDE.U32 q, stride, base
// so the magic-chain walk in IntDivRecognize descends PAST the true IMAD.HI
// quotient leaf into that address IMAD.WIDE.U32 (non-zero addend = base ptr,
// live LO word) and SKIPS it (addend_zero=False lo_dead=False reg_dividend=
// True) — leaving the divide UNrecognized => the float-rcp chain lifts
// faithfully => OFF-BY quotients (dual-gemm: wrong coords -> wrong / duplicate
// write addresses; on the CPU software rcp_f32 the quotient can be WILDLY off).
//
// out[clamp(off)] = in[index] proves coord0=index/N and coord1=index%N are
// exact: a correct divide is a BIJECTION index<->off over a [rows x stride]
// region (stride >= N), off = (index/N)*stride + (index%N). The clamp(off)
// (min against cap-1) keeps the store IN-BOUNDS even when the buggy quotient
// is garbage — so the bug shows as VALUE mismatches / unwritten cells, never a
// segfault. The clamp is one ISETP+SEL; it does NOT add a second divide, so the
// magic chain + IMAD.WIDE.U32 address leaf are emitted exactly as in dual_gemm.
extern "C" __global__ void kern_wideleaf(int* out, const int* in,
                                         unsigned N, long stride,
                                         unsigned total, long cap) {
    unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total) return;
    unsigned coord0 = index / N;     // magic-rcp quotient (vector dividend)
    unsigned coord1 = index % N;     // remainder leaf shares the magic
    long off = (long)coord0 * stride + coord1;  // IMAD.WIDE.U32 q*stride+base
    long soff = off < 0 ? 0 : (off >= cap ? cap - 1 : off);  // clamp (no divide)
    out[soff] = in[index];
}
