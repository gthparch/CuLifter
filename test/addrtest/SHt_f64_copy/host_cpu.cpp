#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void SHt_f64_copy();

int main() {
    constexpr int N = 1024, BS = 256;
    double *h_in = new double[N], *h_out = new double[N];
    memset(h_out, 0, N*sizeof(double));
    for (int i = 0; i < N; i++) h_in[i] = (double)(i*3+7);
    launchKernel(SHt_f64_copy, (N+BS-1)/BS, BS, (double*)h_in, h_out, N);
    bool ok = true;
    for (int i = 0; i < N; i++)
        if (fabs(h_out[i]-h_in[i])>1e-10) { std::cout << "FAIL at " << i << std::endl; ok=false; break; }
    if (ok) std::cout << "TEST PASSED" << std::endl;
    delete[] h_in; delete[] h_out;
    return ok ? 0 : 1;
}
