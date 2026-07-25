#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

#include "cuda_driver_helpers.h"

namespace {

constexpr char kKernelName[] = "_Z27minimal_address_calculationPfi";

struct ProblemConfig {
    int n = 128;
    int block_dim = 32;

    int grid_dim() const {
        return (n + block_dim - 1) / block_dim;
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

    std::vector<float> c(cfg.n, -1.0f);

    CuDriverSession session;
    CUfunction fn = session.loadKernel(cubin_path, kKernelName);

    CUdeviceptr d_c{};
    const size_t bytes = static_cast<size_t>(cfg.n) * sizeof(float);
    CUCHK(cuMemAlloc(&d_c, bytes));
    CUCHK(cuMemcpyHtoD(d_c, c.data(), bytes));

    void* params[] = {&d_c, &cfg.n};
    CUCHK(cuLaunchKernel(
        fn,
        cfg.grid_dim(), 1, 1,
        cfg.block_dim, 1, 1,
        0,
        nullptr,
        params,
        nullptr));
    CUCHK(cuCtxSynchronize());

    CUCHK(cuMemcpyDtoH(c.data(), d_c, bytes));

    bool ok = true;
    for (int i = 0; i < cfg.n; ++i) {
        int block_id = i / cfg.block_dim;
        int thread_id = i % cfg.block_dim;
        int global_id = block_id * cfg.block_dim + thread_id;

        if (global_id < cfg.n) {
            float expected = static_cast<float>(global_id);
            if (std::fabs(c[i] - expected) > 1e-6f) {
                std::cerr << "FAIL at index " << i << ": got " << c[i]
                          << ", expected " << expected << '\n';
                ok = false;
                break;
            }
        }
    }

    if (ok) {
        std::cout << "TEST PASSED\n";
    }

    CUCHK(cuMemFree(d_c));

    return ok ? 0 : 1;
}

