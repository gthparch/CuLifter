#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include "cuda_driver_helpers.h"

namespace {

constexpr char kKernelName[] = "_Z19tile4_reduce_kernelPKfPfi";
constexpr int kTileSize = 4;

struct ProblemConfig {
    int n = 1024;
    int block_dim = 128;

    int grid_dim() const {
        return (n + block_dim - 1) / block_dim;
    }

    int tile_count() const {
        return (n + kTileSize - 1) / kTileSize;
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

    std::vector<float> input(cfg.n);
    std::vector<float> output(cfg.tile_count(), 0.0f);

    std::random_device dev;
    std::mt19937 gen(dev());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < cfg.n; ++i) {
        input[i] = dist(gen);
    }

    CuDriverSession session;
    CUfunction fn = session.loadKernel(cubin_path, kKernelName);

    CUdeviceptr d_input{};
    CUdeviceptr d_output{};

    const size_t input_bytes = static_cast<size_t>(cfg.n) * sizeof(float);
    const size_t output_bytes = static_cast<size_t>(cfg.tile_count()) * sizeof(float);
    CUCHK(cuMemAlloc(&d_input, input_bytes));
    CUCHK(cuMemAlloc(&d_output, output_bytes));

    CUCHK(cuMemcpyHtoD(d_input, input.data(), input_bytes));
    CUCHK(cuMemsetD8(d_output, 0, output_bytes));
    
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

    CUCHK(cuMemcpyDtoH(output.data(), d_output, output_bytes));

    bool ok = true;
    for (int tile = 0; tile < cfg.tile_count(); ++tile) {
        const int start = tile * kTileSize;
        const int end = std::min(cfg.n, start + kTileSize);
        float expected = 0.0f;
        for (int i = start; i < end; ++i) {
            expected += input[i];
        }

        if (std::fabs(output[tile] - expected) > 1e-5f) {
            std::cerr << "Mismatch at tile " << tile << ": got " << output[tile]
                      << ", expected " << expected << "\n";
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
