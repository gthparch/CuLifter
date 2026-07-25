// host_cpu.cpp — driver for F2fp_mergec_rz.
//
// Exercises the dominant CUTLASS fp8 idiom F2FP.E4M3.F32.PACK_AB_MERGE_C
// Rd, RZ, Rb, Rd.H1 (A=RZ shift-register form, as in 88_hopper_fmha_fp8),
// followed by LOP3 byte-masks + PRMT reassembly into an fp8x4. Validates that
// each converted fp8 byte lands in [7:0] (the layout the LOP3/PRMT expect).
//
// CPU oracle: portable software fp8 encoder (fp8_ref.h), e4m3.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include "kernel_wrapper.h"
#include "fp8_ref.h"

extern "C" void F2fp_mergec_rz();

int main() {
    constexpr int N = 64;
    float* in  = new float[N * 4];
    uint32_t* out = new uint32_t[N];
    static const float pool[] = {
        0.0f, -0.0f, 1.0f, -1.0f, 0.5f, -0.25f, 3.14159f, -2.71828f,
        100.0f, -100.0f, 448.0f, 500.0f, -500.0f, 1e-3f, -1e-4f, 256.0f,
        0.015625f, -0.001953125f, 7.0f, 13.0f, 200.0f, 6.1e-5f, -3.0f, 2.0f
    };
    int M = (int)(sizeof(pool)/sizeof(pool[0]));
    for (int i = 0; i < N * 4; i++) in[i] = pool[i % M];
    for (int i = 0; i < N; i++) out[i] = 0xDEADBEEF;

    launchKernel(F2fp_mergec_rz, 1, N,
                 ArgAt{0x210}, in, ArgAt{0x218}, out, ArgAt{0x220}, (int)N);

    int bad = 0;
    for (int i = 0; i < N; i++) {
        uint32_t exp = (uint32_t)f32_to_fp8_ref(in[i*4+0], 0)
                     | ((uint32_t)f32_to_fp8_ref(in[i*4+1], 0) << 8)
                     | ((uint32_t)f32_to_fp8_ref(in[i*4+2], 0) << 16)
                     | ((uint32_t)f32_to_fp8_ref(in[i*4+3], 0) << 24);
        if (out[i] != exp) {
            if (bad < 10)
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
