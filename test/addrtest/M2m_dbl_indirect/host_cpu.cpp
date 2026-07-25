#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void M2m_dbl_indirect();

int main() {
    constexpr int N = 1024, NTABLES = 2, NPTRS = 3;
    constexpr int blockDim = 256;
    constexpr int gridDim = (N + blockDim - 1) / blockDim;

    // Allocate source buffers
    float *bufs[NTABLES][NPTRS];
    for (int t = 0; t < NTABLES; t++)
        for (int p = 0; p < NPTRS; p++) {
            bufs[t][p] = new float[N];
            for (int i = 0; i < N; i++)
                bufs[t][p][i] = (float)((i % 7 + 1 + t + p) * 0.1f);
        }

    // Build pointer arrays (table -> ptrs -> buffers)
    const float* ptr_rows[NTABLES][NPTRS];
    for (int t = 0; t < NTABLES; t++)
        for (int p = 0; p < NPTRS; p++)
            ptr_rows[t][p] = bufs[t][p];

    const float* const* tables[NTABLES];
    for (int t = 0; t < NTABLES; t++)
        tables[t] = ptr_rows[t];

    float *out = new float[N];
    std::memset(out, 0, N * sizeof(float));

    int table = 0, which = 1;
    launchKernel(M2m_dbl_indirect, gridDim, blockDim,
                 (const float* const* const*)tables, out, N, table, which);

    bool ok = true;
    for (int i = 0; i < N; i++) {
        if (out[i] != bufs[table][which][i]) {
            std::cout << "FAIL at " << i << ": got " << out[i]
                      << " expected " << bufs[table][which][i] << "\n";
            ok = false;
            break;
        }
    }
    std::cout << (ok ? "TEST PASSED" : "FAIL") << std::endl;

    for (int t = 0; t < NTABLES; t++)
        for (int p = 0; p < NPTRS; p++)
            delete[] bufs[t][p];
    delete[] out;
    return ok ? 0 : 1;
}
