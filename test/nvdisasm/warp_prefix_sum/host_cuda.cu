#include <algorithm>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include "cuda_driver_helpers.h"

namespace {

constexpr char kKernelName[] = "_Z15warp_prefix_sumPKiPii";

struct ProblemConfig {
    int n = 1024;
    int block_dim = 128;

    int grid_dim() const {
        return (n + block_dim - 1) / block_dim;
    }

    int warp_count() const {
        return (grid_dim() * block_dim + 31) / 32;
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

    std::vector<int> input(cfg.n, 0);
    std::vector<int> output(cfg.n, 0);

    std::random_device dev;
    std::mt19937 gen(dev());
    std::uniform_int_distribution<int> dist(-4, 4);
    for (int i = 0; i < cfg.n; ++i) {
        input[i] = dist(gen);
    }

    CuDriverSession session;
    CUfunction fn = session.loadKernel(cubin_path, kKernelName);

    CUdeviceptr d_input{};
    CUdeviceptr d_output{};

    const size_t bytes = static_cast<size_t>(cfg.n) * sizeof(int);
    CUCHK(cuMemAlloc(&d_input, bytes));
    CUCHK(cuMemAlloc(&d_output, bytes));

    CUCHK(cuMemcpyHtoD(d_input, input.data(), bytes));
    CUCHK(cuMemsetD8(d_output, 0, bytes));

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
    for (int warp = 0; warp < cfg.warp_count() && ok; ++warp) {
        const int start = warp * 32;
        const int end = std::min(cfg.n, start + 32);
        int running = 0;
        for (int idx = start; idx < end; ++idx) {
            running += input[idx];
            if (output[idx] != running) {
                std::cerr << "FAIL at index " << idx
                          << ": got " << output[idx]
                          << ", expected " << running << '\n';
                ok = false;
                break;
            }
        }
    }

    if (ok) {
        std::cout << "TEST PASSED\n";
    }

    CUCHK(cuMemFree(d_input));
    CUCHK(cuMemFree(d_output));

    return ok ? 0 : 1;
}
