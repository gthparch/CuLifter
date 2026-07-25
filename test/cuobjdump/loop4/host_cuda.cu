#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include "cuda_driver_helpers.h"

namespace {

constexpr char kKernelName[] = "_Z5loop4PKfS0_Pfi";

struct ProblemConfig {
    int n = 1024;
    int block_dim = 128;

    int grid_dim() const {
        return (n + block_dim - 1) / block_dim;
    }

    int total_threads() const {
        return grid_dim() * block_dim;
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

    const size_t vec_bytes = static_cast<size_t>(cfg.n) * sizeof(float);
    CUCHK(cuMemAlloc(&d_a, vec_bytes));
    CUCHK(cuMemAlloc(&d_b, vec_bytes));
    CUCHK(cuMemAlloc(&d_c, vec_bytes));

    CUCHK(cuMemcpyHtoD(d_a, a.data(), vec_bytes));
    CUCHK(cuMemcpyHtoD(d_b, b.data(), vec_bytes));
    CUCHK(cuMemsetD8(d_c, 0, vec_bytes));
    
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

    CUCHK(cuMemcpyDtoH(c.data(), d_c, vec_bytes));

    bool ok = true;
    const int total_threads = cfg.total_threads();
    for (int tid = 0; tid < cfg.n; ++tid) {
        float expected = 0;
        for (int i = tid; i < cfg.n; i += 1)
            expected += a[i] + b[i];

        float tol = 1e-3 * std::fabs(expected);
        if (std::fabs(c[tid] - expected) > tol) {
            std::cerr << "Mismatch at " << tid << ": got " << c[tid]
                      << ", expected " << expected << "\n";
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

