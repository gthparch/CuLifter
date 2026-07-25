#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void SH2_neighbor_shared();

int main() {
    constexpr int N = 1024, BS = 256;
    float *h_in = new float[N], *h_out = new float[N];
    memset(h_out, 0, N*sizeof(float));
    for (int i = 0; i < N; i++) h_in[i] = (float)((i%7)+1);
    launchKernel(SH2_neighbor_shared, (N+BS-1)/BS, BS, h_in, h_out, N);
    bool ok = true;
    for (int blk = 0; blk < (N+BS-1)/BS; blk++) {
        for (int t = 1; t < BS - 1; t++) {
            int i = blk * BS + t;
            if (i >= 1 && i < N - 1) {
                float exp = h_in[i-1] + h_in[i] + h_in[i+1];
                if (fabsf(h_out[i] - exp) > 1e-5f) {
                    std::cout << "FAIL at " << i << std::endl; ok = false; goto done;
                }
            }
        }
    }
    done:
    if (ok) std::cout << "TEST PASSED" << std::endl;
    delete[] h_in; delete[] h_out;
    return ok ? 0 : 1;
}
