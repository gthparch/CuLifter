// host_cpu.cpp — driver for F2fp_mergec_h1.
//
// Exercises F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C (with a .H1 C-operand
// half-select) — the SASS lowering of the PTX `cvt.rn.satfinite.e4m3x2.f32`
// fp8x4 pack idiom. Kernel: out[i] = bit pattern of __nv_fp8x4_e4m3(p[0..3]).
//
// CPU oracle (fp8_ref.h) reproduces the canonical PTX layout:
//   byte0 = fp8(p0), byte1 = fp8(p1), byte2 = fp8(p2), byte3 = fp8(p3)
// with RN-to-even satfinite rounding — bit-exact with cuda_fp8's software path.
//
// sm_90 arg layout (from SASS): in=c[0x0][0x210], out=c[0x0][0x218], N=c[0x0][0x220].
#include <cstdio>
#include <cstdint>
#include <cstring>
#include "kernel_wrapper.h"
#include "fp8_ref.h"

extern "C" void F2fp_mergec_h1();

int main() {
    constexpr int N = 64;
    float* in  = new float[N * 4];
    uint32_t* out = new uint32_t[N];
    for (int i = 0; i < N * 4; i++) {
        float base = (i % 13) * 0.5f - 3.0f;      // -3.0 .. +3.0
        if (i % 4 == 3) base = (i % 7) * 60.0f;    // larger, exercises high exp
        in[i] = base;
    }
    for (int i = 0; i < N; i++) out[i] = 0xDEADBEEF;

    launchKernel(F2fp_mergec_h1, 1, N,
                 ArgAt{0x210}, in, ArgAt{0x218}, out, ArgAt{0x220}, (int)N);

    int bad = 0;
    for (int i = 0; i < N; i++) {
        uint32_t exp = (uint32_t)f32_to_fp8_ref(in[i*4+0], 0)
                     | ((uint32_t)f32_to_fp8_ref(in[i*4+1], 0) << 8)
                     | ((uint32_t)f32_to_fp8_ref(in[i*4+2], 0) << 16)
                     | ((uint32_t)f32_to_fp8_ref(in[i*4+3], 0) << 24);
        if (out[i] != exp) {
            if (bad < 8)
                printf("MISMATCH i=%d: lifted=0x%08x expected=0x%08x  (p=%g,%g,%g,%g)\n",
                       i, out[i], exp, in[i*4+0], in[i*4+1], in[i*4+2], in[i*4+3]);
            bad++;
        }
    }
    printf("bad=%d / %d\n", bad, N);
    printf(bad == 0 ? "TEST PASSED\n" : "TEST FAILED\n");
    delete[] in; delete[] out;
    return bad ? 1 : 0;
}
