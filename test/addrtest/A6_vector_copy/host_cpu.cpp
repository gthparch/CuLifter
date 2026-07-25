#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void A6_vector_copy();

// float4 equivalent for CPU: 4 consecutive floats (16 bytes)
struct float4 { float x, y, z, w; };

int main() {
    constexpr int N = 256;  // N float4s = 1024 floats
    constexpr int blockDim = 256;
    constexpr int gridDim = (N + blockDim - 1) / blockDim;

    float4 *in = new float4[N];
    float4 *out = new float4[N];
    std::memset(out, 0, N * sizeof(float4));
    for (int i = 0; i < N; i++) {
        in[i] = {i * 0.1f, i * 0.2f, i * 0.3f, i * 0.4f};
    }

    launchKernel(A6_vector_copy, gridDim, blockDim,
                 (const float4*)in, out, N);

    bool ok = true;
    for (int i = 0; i < N; i++) {
        if (out[i].x != in[i].x || out[i].y != in[i].y ||
            out[i].z != in[i].z || out[i].w != in[i].w) {
            std::cout << "FAIL at " << i << "\n";
            ok = false;
            break;
        }
    }
    std::cout << (ok ? "TEST PASSED" : "FAIL") << std::endl;

    delete[] in; delete[] out;
    return ok ? 0 : 1;
}
