#include <cstdint>
#include <iostream>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void WS5_ballot_popc();

int main() {
    constexpr int N = 256;
    constexpr int blockDim = 256;
    constexpr int gridDim = 1;

    int *in = new int[N];
    int *out = new int[N];
    std::memset(out, 0, N * sizeof(int));
    for (int i = 0; i < N; i++) in[i] = (i % 3 == 0) ? 1 : -1;

    launchKernel(WS5_ballot_popc, gridDim, blockDim,
                 (const int*)in, out, N);

    // Each warp: count how many lanes have val > 0
    bool ok = true;
    for (int w = 0; w < N / 32; w++) {
        int cnt = 0;
        for (int i = w * 32; i < (w + 1) * 32; i++) {
            if (in[i] > 0) cnt++;
        }
        for (int i = w * 32; i < (w + 1) * 32; i++) {
            if (out[i] != cnt) {
                std::cout << "FAIL at " << i << ": got " << out[i]
                          << " expected " << cnt << "\n";
                ok = false;
                goto done;
            }
        }
    }
done:
    std::cout << (ok ? "TEST PASSED" : "FAIL") << std::endl;

    delete[] in; delete[] out;
    return ok ? 0 : 1;
}
