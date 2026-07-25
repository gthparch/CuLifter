#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void X2m_radius4();

int main() {
    constexpr int N = 1024, NPTRS = 2;

    float *bufs[NPTRS];
    for (int p = 0; p < NPTRS; p++) {
        bufs[p] = new float[N];
        for (int i = 0; i < N; i++)
            bufs[p][i] = (float)((i % 7 + 1 + p) * 0.1f);
    }

    const float* ptrs[NPTRS];
    for (int p = 0; p < NPTRS; p++) ptrs[p] = bufs[p];

    float *out = new float[N];
    std::memset(out, 0, N * sizeof(float));

    int which = 0;
    Dim3 grid{(N + 255) / 256, 1};
    Dim3 block{256, 1};
    launchKernel2D(X2m_radius4, grid, block,
                   (const float* const*)ptrs, out, N, which);

    bool ok = true;
    for (int i = 4; i < N - 4; i++) {
        float expected = bufs[which][i-4] + bufs[which][i-3]
                       + bufs[which][i-2] + bufs[which][i-1]
                       + bufs[which][i]
                       + bufs[which][i+1] + bufs[which][i+2]
                       + bufs[which][i+3] + bufs[which][i+4];
        if (std::fabs(out[i] - expected) > 1e-5f) {
            std::cout << "FAIL at " << i << ": got " << out[i]
                      << " expected " << expected << "\n";
            ok = false;
            break;
        }
    }
    std::cout << (ok ? "TEST PASSED" : "FAIL") << std::endl;

    for (int p = 0; p < NPTRS; p++) delete[] bufs[p];
    delete[] out;
    return ok ? 0 : 1;
}
