#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void SHa_broadcast();

int main() {
    constexpr int N = 1024, BS = 256;
    float *h_in = new float[N], *h_out = new float[N];
    memset(h_out, 0, N*sizeof(float));
    for (int i = 0; i < N; i++) h_in[i] = (float)(i*3+7);
    launchKernel(SHa_broadcast, (N+BS-1)/BS, BS, h_in, h_out, N);
    bool ok = true;
    for (int blk = 0; blk < (N+BS-1)/BS; blk++) {
        float expected = h_in[blk*BS];
        for (int t = 0; t < BS; t++) {
            int i = blk*BS+t; if (i>=N) continue;
            if (h_out[i] != expected) {
                std::cout << "FAIL at " << i << std::endl; ok=false; goto done;
            }
        }
    }
    done:
    if (ok) std::cout << "TEST PASSED" << std::endl;
    delete[] h_in; delete[] h_out;
    return ok ? 0 : 1;
}
