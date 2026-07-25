#include <cstdio>
#include <cmath>
#include "kernel_wrapper.h"
// Microbench: DEFINED in-cubin device function called INDIRECTLY through a
// c[0x4] (.nv.constant4) function-pointer slot — the FlashAttnFwdSm90 dispatch
// shape. Without the ResolveIndirectCall defined-symbol fix, the CALL.ABS.NOINC
// falls through to a BPT.TRAP -> llvm.trap -> ud2 -> SIGSEGV (the flash crash).
// Oracle: helper_mul3(x) = x * 3.0f, so y[i] == x[i] * 3.
extern "C" void const4_devcall();
int main() {
    const int N = 8;
    static float x[N] = {1,2,3,4,5,6,7,8};
    static float y[N];
    int bad = 0;
    for (int i = 0; i < N; i++) y[i] = -777;
    *((uint64_t*)(const_mem[0] + 0x210)) = (uint64_t)(uintptr_t)x;   // const float* x
    *((uint64_t*)(const_mem[0] + 0x218)) = (uint64_t)(uintptr_t)y;   // float* y
    *((uint32_t*)(const_mem[0] + 0x220)) = N;                        // int n
    launchKernel(const4_devcall, 1, N);
    for (int i = 0; i < N; i++) {
        float ref = x[i] * 3.0f;
        if (y[i] != ref) {
            if (bad < 3) printf("MISMATCH i=%d got=%g ref=%g\n", i, y[i], ref);
            bad++;
        }
    }
    printf(bad ? "FAIL %d\n" : "TEST PASSED\n", bad);
    return bad != 0;
}
