#include <algorithm>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include "cuda_driver_helpers.h"

namespace {

constexpr char kKernelName[] = "_Z16warp_any_nonzeroPKiPii";

struct ProblemConfig {
    int n = 1024;
    int block_dim = 128;

    int grid_dim() const {
        return (n + block_dim - 1) / block_dim;
    }

    int warp_count() const {
        return (grid_dim() * block_dim) / 32;
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

    std::vector<int> flags(cfg.n, 0);
    std::vector<int> warp_any(cfg.warp_count(), 0);

    std::random_device dev;
    std::mt19937 gen(dev());
    std::uniform_int_distribution<int> dist(-2, 2);
    for (int i = 0; i < cfg.n; ++i) {
        flags[i] = dist(gen);
    }

    CuDriverSession session;
    CUfunction fn = session.loadKernel(cubin_path, kKernelName);

    CUdeviceptr d_flags{};
    CUdeviceptr d_warp_any{};

    const size_t flag_bytes = static_cast<size_t>(cfg.n) * sizeof(int);
    const size_t warp_any_bytes = static_cast<size_t>(cfg.warp_count()) * sizeof(int);
    CUCHK(cuMemAlloc(&d_flags, flag_bytes));
    CUCHK(cuMemAlloc(&d_warp_any, warp_any_bytes));

    CUCHK(cuMemcpyHtoD(d_flags, flags.data(), flag_bytes));
    CUCHK(cuMemsetD8(d_warp_any, 0, warp_any_bytes));

    void* params[] = {&d_flags, &d_warp_any, &cfg.n};
    CUCHK(cuLaunchKernel(
        fn,
        cfg.grid_dim(), 1, 1,
        cfg.block_dim, 1, 1,
        0,
        nullptr,
        params,
        nullptr));
    CUCHK(cuCtxSynchronize());

    CUCHK(cuMemcpyDtoH(warp_any.data(), d_warp_any, warp_any_bytes));

    bool ok = true;
    for (int warp = 0; warp < cfg.warp_count(); ++warp) {
        const int start = warp * 32;
        const int end = std::min(cfg.n, start + 32);
        int expected = 0;
        for (int i = start; i < end; ++i) {
            if (flags[i] != 0) {
                expected = 1;
                break;
            }
        }
        if (warp_any[warp] != expected) {
            std::cerr << "FAIL at warp " << warp
                      << ": got " << warp_any[warp]
                      << ", expected " << expected << '\n';
            ok = false;
            break;
        }
    }

    if (ok) {
        std::cout << "TEST PASSED\n";
    }

    CUCHK(cuMemFree(d_flags));
    CUCHK(cuMemFree(d_warp_any));

    return ok ? 0 : 1;
}
