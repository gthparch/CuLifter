#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void H7_nested_64_affine();

int main() {
    constexpr int N = 1024;

    float *in = new float[N];
    float *out = new float[N];
    std::memset(out, 0, N * sizeof(float));
    for (int i = 0; i < N; i++) in[i] = (float)(i % 7 + 1) * 0.1f;

    unsigned long long S0 = 1048576ULL, S1 = 65536ULL;
    launchKernel(H7_nested_64_affine, 1, 256,
                 (const float*)in, out, S0, S1, N);

    bool ok = true;
    for (int i = 0; i < 256; i++) {
        unsigned long long idx = ((0ULL * 1048576ULL + (unsigned long long)i) * 65536ULL + (unsigned long long)i);
        int j = (int)(idx % 1024ULL);
        if (out[j] != in[j]) {
            std::cout << "FAIL at " << j << ": got " << out[j]
                      << " expected " << in[j] << "\n";
            ok = false;
            break;
        }
    }
    std::cout << (ok ? "TEST PASSED" : "FAIL") << std::endl;

    delete[] in; delete[] out;
    return ok ? 0 : 1;
}
