#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void X7_packed_indirect();

#pragma pack(push, 1)
struct PackedEntry { char tag; float* ptr; short len; };
#pragma pack(pop)

int main() {
    constexpr int N = 256;
    constexpr int blockDim = 256;
    constexpr int gridDim = (N + blockDim - 1) / blockDim;

    float *data = new float[N];
    for (int i = 0; i < N; i++) data[i] = (float)(i % 7 + 1) * 0.1f;

    PackedEntry entries[2] = {{'A', data, (short)N}, {'B', data, (short)N}};

    float *out = new float[N];
    std::memset(out, 0, N * sizeof(float));

    int which = 0;
    launchKernel(X7_packed_indirect, gridDim, blockDim,
                 (const PackedEntry*)entries, out, N, which);

    bool ok = true;
    for (int i = 0; i < N; i++) {
        if (out[i] != data[i]) {
            std::cout << "FAIL at " << i << ": got " << out[i]
                      << " expected " << data[i] << "\n";
            ok = false;
            break;
        }
    }
    std::cout << (ok ? "TEST PASSED" : "FAIL") << std::endl;

    delete[] data; delete[] out;
    return ok ? 0 : 1;
}
