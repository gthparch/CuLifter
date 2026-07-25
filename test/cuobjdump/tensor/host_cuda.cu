#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include <cuda_fp16.h>

#include "cuda_driver_helpers.h"

namespace {

constexpr char kKernelName[] = "_Z9wmma_gemmP6__halfS0_Pfiii";

struct ProblemConfig {
    int m = 64;
    int n = 64;
    int k = 64;
    int block_dim = 128;

    int total_warps() const {
        const int warp_m = (m + 15) / 16;
        const int warp_n = (n + 15) / 16;
        return warp_m * warp_n;
    }

    int grid_dim() const {
        const int warps_per_block = block_dim / 32;
        return (total_warps() + warps_per_block - 1) / warps_per_block;
    }
};

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " <cubin|ptx>\n";
        return 1;
    }

    const char* device_img_path = argv[1];
    ProblemConfig cfg;

    std::vector<__half> a(static_cast<size_t>(cfg.m) * cfg.k);
    std::vector<__half> b(static_cast<size_t>(cfg.k) * cfg.n);
    std::vector<float> c(static_cast<size_t>(cfg.m) * cfg.n, 0.0f);
    std::vector<float> ref(static_cast<size_t>(cfg.m) * cfg.n, 0.0f);

    std::mt19937 gen(12345);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < a.size(); ++i) {
        a[i] = __float2half(dist(gen));
    }
    for (size_t i = 0; i < b.size(); ++i) {
        b[i] = __float2half(dist(gen));
    }

    CuDriverSession session;
    CUfunction fn = session.loadKernel(device_img_path, kKernelName);

    CUdeviceptr d_a{};
    CUdeviceptr d_b{};
    CUdeviceptr d_c{};

    const size_t a_bytes = a.size() * sizeof(__half);
    const size_t b_bytes = b.size() * sizeof(__half);
    const size_t c_bytes = c.size() * sizeof(float);

    CUCHK(cuMemAlloc(&d_a, a_bytes));
    CUCHK(cuMemAlloc(&d_b, b_bytes));
    CUCHK(cuMemAlloc(&d_c, c_bytes));

    CUCHK(cuMemcpyHtoD(d_a, a.data(), a_bytes));
    CUCHK(cuMemcpyHtoD(d_b, b.data(), b_bytes));
    CUCHK(cuMemsetD8(d_c, 0, c_bytes));

    void* params[] = {&d_a, &d_b, &d_c, &cfg.m, &cfg.n, &cfg.k};
    CUCHK(cuLaunchKernel(
        fn,
        cfg.grid_dim(), 1, 1,
        cfg.block_dim, 1, 1,
        0,
        nullptr,
        params,
        nullptr));
    CUCHK(cuCtxSynchronize());

    CUCHK(cuMemcpyDtoH(c.data(), d_c, c_bytes));

    bool ok = true;
    for (int i = 0; i < cfg.m; ++i) {
        for (int j = 0; j < cfg.n; ++j) {
            float acc = 0.0f;
            for (int kk = 0; kk < cfg.k; ++kk) {
                acc += __half2float(a[static_cast<size_t>(i) * cfg.k + kk]) *
                       __half2float(b[static_cast<size_t>(kk) * cfg.n + j]);
            }
            ref[static_cast<size_t>(i) * cfg.n + j] = acc;
        }
    }

    constexpr float kTolerance = 1e-1f;
    for (size_t idx = 0; idx < c.size(); ++idx) {
        if (std::fabs(c[idx] - ref[idx]) > kTolerance) {
            std::cerr << "FAIL at index " << idx << ": got " << c[idx]
                      << ", expected " << ref[idx] << '\n';
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
