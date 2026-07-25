#include <cstdint>
#include <iostream>
#include <cstring>
#include <cstdio>
#include "kernel_wrapper.h"
extern "C" void PHI0_if_else();
int main() {
    constexpr int N = 256;
    int *h_in = new int[N], *h_out = new int[N];
    memset(h_out, 0, N*sizeof(int));
    for (int i = 0; i < N; i++) h_in[i] = i * 7 - 100;
    launchKernel(PHI0_if_else, (N+255)/256, 256, h_in, h_out, N);
    // Compare against ref_output.bin
    FILE* f = fopen("ref_output.bin", "rb");
    if (!f) { std::cerr << "No ref_output.bin" << std::endl; return 1; }
    int ref[N]; fread(ref, sizeof(int), N, f); fclose(f);
    bool ok = true;
    for (int i = 0; i < N; i++)
        if (h_out[i] != ref[i]) { std::cout << "FAIL at " << i << ": got " << h_out[i] << " exp " << ref[i] << std::endl; ok=false; break; }
    if (ok) std::cout << "TEST PASSED" << std::endl;
    delete[] h_in; delete[] h_out;
    return ok ? 0 : 1;
}
