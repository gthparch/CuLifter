#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void SH14_reduce_sum_16();

int main() {
    constexpr int N = 1024, BS = 16, NBLK = (N+BS-1)/BS;
    float *h_in = new float[N], *h_out = new float[NBLK];
    memset(h_out, 0, NBLK*sizeof(float));
    for (int i = 0; i < N; i++) h_in[i] = 1.0f;
    launchKernel(SH14_reduce_sum_16, NBLK, BS, h_in, h_out, N);
    bool ok = true;
    for (int b = 0; b < NBLK; b++) {
        int count = (b < NBLK-1) ? BS : (N - b*BS);
        if (fabsf(h_out[b] - (float)count) > 0.5f) {
            std::cout << "FAIL blk " << b << ": got " << h_out[b] << " exp " << count << std::endl;
            ok = false; break;
        }
    }
    if (ok) std::cout << "TEST PASSED" << std::endl;
    delete[] h_in; delete[] h_out;
    return ok ? 0 : 1;
}
