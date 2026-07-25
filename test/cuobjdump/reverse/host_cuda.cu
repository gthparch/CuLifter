#include <algorithm>
#include <iostream>
#include <vector>

#include "cuda_driver_helpers.h"

namespace {

constexpr char kKernelName[] = "_Z7reversePii";

struct ProblemConfig {
    int len = 256;
    int block_dim = 256;

    int grid_dim() const {
        return (len + block_dim - 1) / block_dim;
    }
};

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " <cubin>\n";
        return 1;
    }

    const char* cubin_path = argv[1];
    ProblemConfig cfg;

    std::vector<int> values(cfg.len);
    for (int i = 0; i < cfg.len; ++i) {
        values[i] = i;
    }

    auto expected = values;
    std::reverse(expected.begin(), expected.end());

    CuDriverSession session;
    CUfunction fn = session.loadKernel(cubin_path, kKernelName);

    CUdeviceptr d_values{};

    const size_t bytes = static_cast<size_t>(cfg.len) * sizeof(int);
    CUCHK(cuMemAlloc(&d_values, bytes));

    CUCHK(cuMemcpyHtoD(d_values, values.data(), bytes));

    void* params[] = {&d_values, &cfg.len};
    CUCHK(cuLaunchKernel(
        fn,
        cfg.grid_dim(), 1, 1,
        cfg.block_dim, 1, 1,
        0,
        nullptr,
        params,
        nullptr));
    CUCHK(cuCtxSynchronize());

    CUCHK(cuMemcpyDtoH(values.data(), d_values, bytes));

    bool ok = true;
    for (int i = 0; i < cfg.len; ++i) {
        if (values[i] != expected[i]) {
            std::cerr << "FAIL at index " << i
                      << ": got " << values[i]
                      << ", expected " << expected[i] << '\n';
            ok = false;
            break;
        }
    }

    if (ok) {
        std::cout << "TEST PASSED\n";
    }

    CUCHK(cuMemFree(d_values));

    return ok ? 0 : 1;
}
