#include <cstdio>
#include <cmath>
#include "kernel_wrapper.h"
extern "C" void indirect_call();
int main() {
    const int N = 8;
    static float x[N] = {1,2,3,4,5,6,7,8};
    static float y[N];
    int bad = 0;
    for (int sel = 0; sel <= 1; sel++) {
        for (int i = 0; i < N; i++) y[i] = -777;
        *((uint64_t*)(const_mem[0] + 0x210)) = (uint64_t)(uintptr_t)x;
        *((uint64_t*)(const_mem[0] + 0x218)) = (uint64_t)(uintptr_t)y;
        *((uint32_t*)(const_mem[0] + 0x220)) = sel;
        *((uint32_t*)(const_mem[0] + 0x224)) = N;
        launchKernel(indirect_call, 1, N);
        for (int i = 0; i < N; i++) {
            float ref = sel ? x[i]*2.0f : x[i]+10.0f;
            if (y[i] != ref) { if (bad<3) printf("MISMATCH sel=%d i=%d got=%g ref=%g\n", sel, i, y[i], ref); bad++; }
        }
    }
    printf(bad ? "FAIL %d\n" : "TEST PASSED\n", bad);
    return bad != 0;
}
