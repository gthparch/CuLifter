#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void L1_array_of_structs();

struct MyStruct { float x; int y; float z; };

int main() {
    constexpr int N = 1024;
    constexpr int blockDim = 256;
    constexpr int gridDim = (N + blockDim - 1) / blockDim;

    MyStruct *arr = new MyStruct[N];
    float *out = new float[N];
    std::memset(out, 0, N * sizeof(float));
    for (int i = 0; i < N; i++) {
        arr[i] = {(float)(i % 7 + 1) * 0.1f, i % 5 + 1, (float)(i % 3 + 1) * 0.2f};
    }

    launchKernel(L1_array_of_structs, gridDim, blockDim,
                 (const MyStruct*)arr, out, N);

    bool ok = true;
    for (int i = 0; i < N; i++) {
        float expected = arr[i].x + (float)arr[i].y + arr[i].z;
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
