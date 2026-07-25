#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void SH15_reduce_max_32();

int main() {
    constexpr int N = 1024, BS = 32, NBLK = (N+BS-1)/BS;
    float *h_in = new float[N], *h_out = new float[NBLK];
    memset(h_out, 0, NBLK*sizeof(float));
    for (int i = 0; i < N; i++) h_in[i] = (float)(i % 100);
    launchKernel(SH15_reduce_max_32, NBLK, BS, h_in, h_out, N);
    bool ok = true;
    for (int b = 0; b < NBLK; b++) {
        float mx = -1e30f;
        for (int t = 0; t < BS && b*BS+t < N; t++)
            if (h_in[b*BS+t] > mx) mx = h_in[b*BS+t];
        if (fabsf(h_out[b] - mx) > 1e-5f) {
            std::cout << "FAIL blk " << b << ": got " << h_out[b] << " exp " << mx << std::endl;
            ok = false; break;
        }
    }
    if (ok) std::cout << "TEST PASSED" << std::endl;
    delete[] h_in; delete[] h_out;
    return ok ? 0 : 1;
}
