#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include "cuda_driver_helpers.h"

namespace {

constexpr char kKernelName[] = "_Z5saxpyifPKfPf";

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

    float a = 2.0f;
    std::vector<float> x(cfg.n, 0.0f);
    std::vector<float> y(cfg.n, 0.0f);

    std::random_device dev;
    std::mt19937 gen(dev());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < cfg.n; ++i) {
        x[i] = dist(gen);
        y[i] = dist(gen);
    }

    std::vector<float> result_y = y;

    CuDriverSession session;
    CUfunction fn = session.loadKernel(cubin_path, kKernelName);

    CUdeviceptr d_x{};
    CUdeviceptr d_y{};

    const size_t bytes = static_cast<size_t>(cfg.n) * sizeof(float);
    CUCHK(cuMemAlloc(&d_x, bytes));
    CUCHK(cuMemAlloc(&d_y, bytes));

    CUCHK(cuMemcpyHtoD(d_x, x.data(), bytes));
    CUCHK(cuMemcpyHtoD(d_y, result_y.data(), bytes));

    void* params[] = {&cfg.n, &a, &d_x, &d_y};
    CUCHK(cuLaunchKernel(
        fn,
        cfg.grid_dim(), 1, 1,
        cfg.block_dim, 1, 1,
        0,
        nullptr,
        params,
        nullptr));
    CUCHK(cuCtxSynchronize());

    CUCHK(cuMemcpyDtoH(result_y.data(), d_y, bytes));

    bool ok = true;
    for (int i = 0; i < cfg.n; ++i) {
        float expected = a * x[i] + y[i];
        if (std::fabs(result_y[i] - expected) > 1e-6f) {
            std::cerr << "FAIL at index " << i
                      << ": got " << result_y[i]
                      << ", expected " << expected << '\n';
            ok = false;
            break;
        }
    }

    if (ok) {
        std::cout << "TEST PASSED\n";
    }

    CUCHK(cuMemFree(d_x));
    CUCHK(cuMemFree(d_y));

    return ok ? 0 : 1;
}
