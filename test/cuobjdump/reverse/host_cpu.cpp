#include <algorithm>
#include <iostream>
#include <vector>

#include "kernel_wrapper.h"

extern "C" void _Z7reversePii();

int main() {
    constexpr int kLen      = 256;
    constexpr int blockDim  = kLen;
    constexpr int gridDim   = 1;

    std::vector<int> data(kLen);
    for (int i = 0; i < kLen; ++i) {
        data[i] = i;
    }

    auto expected = data;
    std::reverse(expected.begin(), expected.end());

    launchKernel(_Z7reversePii, gridDim, blockDim, data.data(), kLen);

    bool ok = true;
    for (int i = 0; i < kLen; ++i) {
        if (data[i] != expected[i]) {
            std::cout << "FAIL at index " << i
                      << ": got " << data[i]
                      << ", expected " << expected[i] << "\n";
            ok = false;
            break;
        }
    }

    if (ok) {
        std::cout << "TEST PASSED\n";
    }

    return ok ? 0 : 1;
}
