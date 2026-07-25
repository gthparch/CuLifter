#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include "cuda_driver_helpers.h"

namespace {

constexpr char kKernelName[] = "_Z4reluPKfPfi";

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

    std::vector<float> input(cfg.n, 0.0f);
    std::vector<float> output(cfg.n, 0.0f);

    std::random_device dev;
    std::mt19937 gen(dev());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < cfg.n; ++i) {
        input[i] = dist(gen);
    }

    CuDriverSession session;
    CUfunction fn = session.loadKernel(cubin_path, kKernelName);

    CUdeviceptr d_input{};
    CUdeviceptr d_output{};

    const size_t bytes = static_cast<size_t>(cfg.n) * sizeof(float);
    CUCHK(cuMemAlloc(&d_input, bytes));
    CUCHK(cuMemAlloc(&d_output, bytes));

    CUCHK(cuMemcpyHtoD(d_input, input.data(), bytes));
    CUCHK(cuMemcpyHtoD(d_output, output.data(), bytes));

    void* params[] = {&d_input, &d_output, &cfg.n};
    CUCHK(cuLaunchKernel(
        fn,
        cfg.grid_dim(), 1, 1,
        cfg.block_dim, 1, 1,
        0,
        nullptr,
        params,
        nullptr));
    CUCHK(cuCtxSynchronize());

    CUCHK(cuMemcpyDtoH(output.data(), d_output, bytes));

    bool ok = true;
    for (int i = 0; i < cfg.n; ++i) {
        float expected = input[i] > 0.0f ? input[i] : 0.0f;
        if (std::fabs(output[i] - expected) > 1e-6f) {
            std::cerr << "FAIL at index " << i
                      << ": got " << output[i]
                      << ", expected " << expected << '\n';
            ok = false;
            break;
        }
    }

    if (ok) {
        std::cout << "TEST PASSED\n";
    }

    CUCHK(cuMemFree(d_input));
    CUCHK(cuMemFree(d_output));

    return ok ? 0 : 1;
}
