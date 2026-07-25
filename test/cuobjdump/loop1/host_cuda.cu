#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include "cuda_driver_helpers.h"

namespace {

constexpr char kKernelName[] = "_Z5loop1PKfS0_Pfi";

struct ProblemConfig {
    int n = 1024;
    int block_dim = 128;

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

    std::vector<float> a(cfg.n);
    std::vector<float> b(cfg.n);
    std::vector<float> c(cfg.n, 0.0f);

    std::random_device dev;
    std::mt19937 gen(dev());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < cfg.n; ++i) {
        a[i] = dist(gen);
        b[i] = dist(gen);
    }

    CuDriverSession session;
    CUfunction fn = session.loadKernel(cubin_path, kKernelName);

    CUdeviceptr d_a{};
    CUdeviceptr d_b{};
    CUdeviceptr d_c{};

    const size_t bytes = static_cast<size_t>(cfg.n) * sizeof(float);
    CUCHK(cuMemAlloc(&d_a, bytes));
    CUCHK(cuMemAlloc(&d_b, bytes));
    CUCHK(cuMemAlloc(&d_c, bytes));

    CUCHK(cuMemcpyHtoD(d_a, a.data(), bytes));
    CUCHK(cuMemcpyHtoD(d_b, b.data(), bytes));
    CUCHK(cuMemsetD8(d_c, 0, bytes));

    void* params[] = {&d_a, &d_b, &d_c, &cfg.n};
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
    for (int idx = 0; idx < cfg.n; ++idx) {
        double expected = (static_cast<double>(cfg.n - 1) + idx) * (cfg.n - idx) * 0.5
                          + static_cast<double>(a[idx]) + static_cast<double>(b[idx]);
        if (std::fabs(static_cast<double>(c[idx]) - expected) > 1e-3 * expected) {
            std::cerr << "Mismatch at " << idx << ": got " << c[idx]
                      << ", expected " << expected << '\n';
            ok = false;
            break;
        }
    }

    if (ok) {
        std::cout << "TEST PASSED\n";
    }

    CUCHK(cuMemFree(d_a));
    CUCHK(cuMemFree(d_b));
    CUCHK(cuMemFree(d_c));

    return ok ? 0 : 1;
}

