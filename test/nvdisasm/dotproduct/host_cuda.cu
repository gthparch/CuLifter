#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include "cuda_driver_helpers.h"

namespace {

constexpr char kKernelName[] = "_Z11dot_productPKfS0_Pfi";

struct ProblemConfig {
    int n = 1024;
    int block_dim = 128;

    int grid_dim() const {
        return (n + block_dim - 1) / block_dim;
    }

    int shared_mem_bytes() const {
        return block_dim * static_cast<int>(sizeof(float));
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

    std::vector<float> A(cfg.n, 0.0f);
    std::vector<float> B(cfg.n, 0.0f);
    std::vector<float> block_sums(cfg.grid_dim(), 0.0f);

    std::random_device dev;
    std::mt19937 gen(dev());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < cfg.n; ++i) {
        A[i] = dist(gen);
        B[i] = dist(gen);
    }

    CuDriverSession session;
    CUfunction fn = session.loadKernel(cubin_path, kKernelName);

    CUdeviceptr d_A{};
    CUdeviceptr d_B{};
    CUdeviceptr d_block_sums{};

    const size_t vec_bytes = static_cast<size_t>(cfg.n) * sizeof(float);
    const size_t block_sum_bytes = static_cast<size_t>(cfg.grid_dim()) * sizeof(float);
    CUCHK(cuMemAlloc(&d_A, vec_bytes));
    CUCHK(cuMemAlloc(&d_B, vec_bytes));
    CUCHK(cuMemAlloc(&d_block_sums, block_sum_bytes));

    CUCHK(cuMemcpyHtoD(d_A, A.data(), vec_bytes));
    CUCHK(cuMemcpyHtoD(d_B, B.data(), vec_bytes));
    CUCHK(cuMemsetD8(d_block_sums, 0, block_sum_bytes));

    void* params[] = {&d_A, &d_B, &d_block_sums, &cfg.n};
    CUCHK(cuLaunchKernel(
        fn,
        cfg.grid_dim(), 1, 1,
        cfg.block_dim, 1, 1,
        cfg.shared_mem_bytes(),
        nullptr,
        params,
        nullptr));
    CUCHK(cuCtxSynchronize());

    CUCHK(cuMemcpyDtoH(block_sums.data(), d_block_sums, block_sum_bytes));

    bool ok = true;
    for (int block = 0; block < cfg.grid_dim(); ++block) {
        const int start = block * cfg.block_dim;
        const int end = std::min(cfg.n, start + cfg.block_dim);
        float expected = 0.0f;
        for (int i = start; i < end; ++i) {
            expected += A[i] * B[i];
        }
        if (std::fabs(block_sums[block] - expected) > 1e-4f) {
            std::cerr << "FAIL at block " << block
                      << ": got " << block_sums[block]
                      << ", expected " << expected << '\n';
            ok = false;
            break;
        }
    }

    if (ok) {
        std::cout << "TEST PASSED\n";
    }

    CUCHK(cuMemFree(d_A));
    CUCHK(cuMemFree(d_B));
    CUCHK(cuMemFree(d_block_sums));

    return ok ? 0 : 1;
}
