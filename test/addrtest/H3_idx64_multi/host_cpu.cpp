#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void H3_idx64_multi();

int main() {
    constexpr int N = 1024;

    float *in = new float[N];
    float *out = new float[N];
    std::memset(out, 0, N * sizeof(float));
    for (int i = 0; i < N; i++) in[i] = (float)(i % 7 + 1) * 0.1f;

    size_t a = 131072ULL, b = 65536ULL, c = 0ULL;
    launchKernel(H3_idx64_multi, 1, 256,
                 (const float*)in, out, a, b, c, N);

    bool ok = true;
    for (int i = 0; i < 256; i++) {
        size_t idx = 131072ULL * i + 0ULL;
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
