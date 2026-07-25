#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void L2_nested_struct();

struct Inner { float a; float b; };
struct Outer { int id; Inner inner; float c; };

int main() {
    constexpr int N = 1024;
    constexpr int blockDim = 256;
    constexpr int gridDim = (N + blockDim - 1) / blockDim;

    Outer *arr = new Outer[N];
    float *out = new float[N];
    std::memset(out, 0, N * sizeof(float));
    for (int i = 0; i < N; i++) {
        arr[i] = {i, {(float)(i % 7 + 1) * 0.1f, (float)(i % 5 + 1) * 0.2f}, (float)(i % 3 + 1) * 0.3f};
    }

    launchKernel(L2_nested_struct, gridDim, blockDim,
                 (const Outer*)arr, out, N);

    bool ok = true;
    for (int i = 0; i < N; i++) {
        float expected = arr[i].inner.a + arr[i].inner.b + arr[i].c;
        if (std::fabs(out[i] - expected) > 1e-5f) {
            std::cout << "FAIL at " << i << ": got " << out[i]
                      << " expected " << expected << "\n";
            ok = false;
            break;
        }
    }
    std::cout << (ok ? "TEST PASSED" : "FAIL") << std::endl;

    delete[] arr; delete[] out;
    return ok ? 0 : 1;
}
