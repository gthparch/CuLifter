#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void L3_packed_struct();

#pragma pack(push, 1)
struct Packed { char a; float b; short c; float d; };
#pragma pack(pop)

int main() {
    constexpr int N = 1024;
    constexpr int blockDim = 256;
    constexpr int gridDim = (N + blockDim - 1) / blockDim;

    Packed *arr = new Packed[N];
    float *out = new float[N];
    std::memset(out, 0, N * sizeof(float));
    for (int i = 0; i < N; i++) {
        arr[i] = {(char)(i % 26 + 'a'), (float)(i % 7 + 1) * 0.1f,
                  (short)(i % 10), (float)(i % 3 + 1) * 0.2f};
    }

    launchKernel(L3_packed_struct, gridDim, blockDim,
                 (const Packed*)arr, out, N);

    bool ok = true;
    for (int i = 0; i < N; i++) {
        float expected = arr[i].b + arr[i].d + (float)arr[i].c;
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
