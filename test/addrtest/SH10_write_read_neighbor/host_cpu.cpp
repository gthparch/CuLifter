#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void SH10_write_read_neighbor();

int main() {
    constexpr int N = 1024, BS = 256;
    float *h_in = new float[N], *h_out = new float[N];
    memset(h_out, 0, N*sizeof(float));
    for (int i = 0; i < N; i++) h_in[i] = (float)(i*3+7);
    launchKernel(SH10_write_read_neighbor, (N+BS-1)/BS, BS, h_in, h_out, N);
    bool ok = true;
    for (int blk = 0; blk < (N+BS-1)/BS; blk++)
        for (int t = 0; t < BS; t++) {
            int i = blk*BS + t;
            if (i >= N) continue;
            int nb = t ^ 1;
            int gi = blk*BS + nb;
            if (gi < N && h_out[i] != h_in[gi]) {
                std::cout << "FAIL at " << i << std::endl; ok=false; goto done;
            }
        }
    done:
    if (ok) std::cout << "TEST PASSED" << std::endl;
    delete[] h_in; delete[] h_out;
    return ok ? 0 : 1;
}
