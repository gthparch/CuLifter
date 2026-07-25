// host_cpu.cpp — driver for F2fp_mergec_h1_e5m2 (E5M2 fp8x4 pack via MERGE_C.H1).
// CPU oracle: portable software fp8 encoder (fp8_ref.h). Exercises subnormals,
// saturation, NaN, and ordinary values.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include "kernel_wrapper.h"
#include "fp8_ref.h"

extern "C" void F2fp_mergec_h1_e5m2();

int main() {
    constexpr int N = 64;
    float* in  = new float[N * 4];
    uint32_t* out = new uint32_t[N];
    static const float pool[] = {
        0.0f, -0.0f, 1.0f, -1.0f, 0.5f, -0.25f, 3.14159f, -2.71828f,
        100.0f, -100.0f, 57344.0f, 70000.0f, -70000.0f, 1e-5f, -1e-7f, 12345.0f,
        0.015625f, -0.0009765625f, 448.0f, 449.0f, 65504.0f, 6.1e-5f, -3.0f, 2.0f
    };
    int M = (int)(sizeof(pool)/sizeof(pool[0]));
    for (int i = 0; i < N * 4; i++) in[i] = pool[i % M];
    for (int i = 0; i < N; i++) out[i] = 0xDEADBEEF;

    launchKernel(F2fp_mergec_h1_e5m2, 1, N,
                 ArgAt{0x210}, in, ArgAt{0x218}, out, ArgAt{0x220}, (int)N);

    int bad = 0;
    for (int i = 0; i < N; i++) {
        uint32_t exp = (uint32_t)f32_to_fp8_ref(in[i*4+0], 1)
                     | ((uint32_t)f32_to_fp8_ref(in[i*4+1], 1) << 8)
                     | ((uint32_t)f32_to_fp8_ref(in[i*4+2], 1) << 16)
                     | ((uint32_t)f32_to_fp8_ref(in[i*4+3], 1) << 24);
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
