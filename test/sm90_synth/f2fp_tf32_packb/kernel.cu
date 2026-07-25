// F2FP.TF32.F32.PACK_B fixture — the sm90 f32 -> tf32 round.
//
// PTX `cvt.rna.tf32.f32` (and `cvt.rn.tf32.f32`) lower to a single-source
// `F2FP.TF32.F32.PACK_B Rd, Rs` on sm_90. tf32 keeps the f32 sign+exponent
// and only the TOP 10 mantissa bits (19-bit format); the LOW 13 mantissa bits
// are rounded away (round-to-nearest, ties handled per the chosen .rn/.rna).
// The bit pattern is stored back in a full 32-bit register with the low 13
// bits ZEROED. CUTLASS emits 480 of these in the sm90 corpus (the
// NumericConverter<tfloat32_t, float> that rounds accumulators / A,B operands
// to tf32 precision before the tensor op or before a store).
//
// This kernel rounds each input f32 to tf32 and stores the resulting 32-bit
// bit pattern, so a non-tensor consumer (the global store) observes the
// rounded value — exactly the case where a passthrough lift is wrong.
//
// We use cvt.rna.tf32.f32 (round-to-nearest, ties-to-away from the PTX ISA's
// tf32 rounding) — that is what nvcc emits for __float2tf32 / the CUTLASS
// converter and what ptxas turns into F2FP.TF32.F32.PACK_B.
#include <cstdint>

extern "C" __global__ void kern_f2fp_tf32_packb(const float* __restrict__ in,
                                                uint32_t* __restrict__ out,
                                                int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float v = in[i];
    uint32_t packed;
    // cvt.rn.tf32.f32 → F2FP.TF32.F32.PACK_B (the corpus form; round-to-nearest-
    // EVEN). The result is a 32-bit reg whose low 13 mantissa bits are zeroed
    // (tf32 precision), upper 19 bits RN-even rounded. (cvt.rna ties-away does
    // NOT fuse — ptxas emits an explicit HFMA2-bias + LOP3-mask sequence there.)
    asm volatile("cvt.rn.tf32.f32 %0, %1;" : "=r"(packed) : "f"(v));
    out[i] = packed;
}
