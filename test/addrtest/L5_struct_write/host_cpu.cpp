#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void L5_struct_write();

struct Vec3 { float x, y, z; };

int main() {
    constexpr int N = 1024;
    constexpr int blockDim = 256;
    constexpr int gridDim = (N + blockDim - 1) / blockDim;

    float *in = new float[N];
    Vec3 *out = new Vec3[N];
    std::memset(out, 0, N * sizeof(Vec3));
    for (int i = 0; i < N; i++) in[i] = (float)(i % 7 + 1) * 0.1f;

    launchKernel(L5_struct_write, gridDim, blockDim,
                 (const float*)in, out, N);

    bool ok = true;
    for (int i = 0; i < N; i++) {
        if (std::fabs(out[i].x - in[i] * 1.0f) > 1e-5f ||
            std::fabs(out[i].y - in[i] * 2.0f) > 1e-5f ||
            std::fabs(out[i].z - in[i] * 3.0f) > 1e-5f) {
            std::cout << "FAIL at " << i << "\n";
            ok = false;
            break;
        }
    }
    std::cout << (ok ? "TEST PASSED" : "FAIL") << std::endl;

    delete[] in; delete[] out;
    return ok ? 0 : 1;
}
