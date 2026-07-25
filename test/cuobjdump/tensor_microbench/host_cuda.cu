// Driver for tensor_microbench.cu: runs all 15 WMMA microbench kernels
// sequentially, prints "TEST PASSED" only if every one matches a CPU
// reference within tolerance, otherwise prints the first failing kernel
// + index/got/expected.
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

#include <cuda_fp16.h>

#include "cuda_driver_helpers.h"

namespace {

struct KernelCase {
    const char* name;
    int M, N, K;
    int grid_dim;
    int block_dim;
    bool runtime_args;  // pass M/N/K as kernel params?
};

// All 15 cases. Dimensions chosen to keep each test small & deterministic.
const KernelCase kCases[] = {
    {"tm_simple",         16, 16, 16,  1,  32, false},
    {"tm_2k",             16, 16, 32,  1,  32, false},
    {"tm_2warp",          16, 32, 16,  1,  64, false},
    {"tm_4warp",          32, 32, 16,  1, 128, false},
    {"tm_4block",         32, 32, 16,  4,  32, false},
    {"tm_runtime",        16, 16, 16,  1,  32, true },
    {"tm_runtime_kne",    16, 16, 32,  1,  32, true },
    {"tm_2blk2warp",      32, 32, 16,  2,  64, false},
    {"tm_4blk4warp",      64, 64, 16,  4, 128, false},
    {"tm_4warp_rt",       32, 32, 16,  1, 128, true },
    {"tm_4blk4warp_rt",   64, 64, 16,  4, 128, true },
    {"tm_kloop_4warp",    32, 32, 64,  1, 128, true },
    {"tm_kloop_grid",     64, 64, 64,  4, 128, true },
    {"tm_full",          256,256,256, 64, 128, true },
    {"tm_kloop_runtime",  64, 64, 64,  4, 128, true },
};

bool run_one(CuDriverSession& session, const char* device_img, const KernelCase& tc,
             std::ostream& err) {
    CUfunction fn;
    try {
        fn = session.loadKernel(device_img, tc.name);
    } catch (...) {
        err << "FAIL " << tc.name << ": kernel not found in cubin\n";
        return false;
    }

    std::vector<__half> a(static_cast<size_t>(tc.M) * tc.K);
    std::vector<__half> b(static_cast<size_t>(tc.K) * tc.N);
    std::vector<float>  c(static_cast<size_t>(tc.M) * tc.N, 0.0f);
    std::vector<float>  ref(static_cast<size_t>(tc.M) * tc.N, 0.0f);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (auto& x : a) x = __float2half(dist(gen));
    for (auto& x : b) x = __float2half(dist(gen));

    CUdeviceptr d_a{}, d_b{}, d_c{};
    CUCHK(cuMemAlloc(&d_a, a.size() * sizeof(__half)));
    CUCHK(cuMemAlloc(&d_b, b.size() * sizeof(__half)));
    CUCHK(cuMemAlloc(&d_c, c.size() * sizeof(float)));
    CUCHK(cuMemcpyHtoD(d_a, a.data(), a.size() * sizeof(__half)));
    CUCHK(cuMemcpyHtoD(d_b, b.data(), b.size() * sizeof(__half)));
    CUCHK(cuMemsetD8(d_c, 0, c.size() * sizeof(float)));

    int M = tc.M, N = tc.N, K = tc.K;
    void* params_const[] = {&d_a, &d_b, &d_c};
    void* params_runtime[] = {&d_a, &d_b, &d_c, &M, &N, &K};
    void** params = tc.runtime_args ? params_runtime : params_const;

    CUCHK(cuLaunchKernel(fn,
                         tc.grid_dim, 1, 1,
                         tc.block_dim, 1, 1,
                         0, nullptr, params, nullptr));
    CUCHK(cuCtxSynchronize());
    CUCHK(cuMemcpyDtoH(c.data(), d_c, c.size() * sizeof(float)));

    float tol = (tc.K >= 64 ? 5e-1f : (tc.K >= 32 ? 2e-1f : 1e-1f));
    if (tc.M >= 256) tol = 1.0f;  // wider tolerance for the large-K accumulation

    for (int i = 0; i < tc.M; ++i) {
        for (int j = 0; j < tc.N; ++j) {
            float acc = 0.0f;
            for (int kk = 0; kk < tc.K; ++kk) {
                acc += __half2float(a[i * tc.K + kk]) *
                       __half2float(b[kk * tc.N + j]);
            }
            ref[i * tc.N + j] = acc;
        }
    }

    bool ok = true;
    for (size_t idx = 0; idx < c.size(); ++idx) {
        if (std::fabs(c[idx] - ref[idx]) > tol) {
            err << "FAIL " << tc.name << " at idx " << idx
                << ": got " << c[idx] << ", expected " << ref[idx] << "\n";
            ok = false;
            break;
        }
    }

    CUCHK(cuMemFree(d_a));
    CUCHK(cuMemFree(d_b));
    CUCHK(cuMemFree(d_c));
    return ok;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " <cubin|ptx> [kernel_name_filter]\n";
        return 1;
    }
    const char* device_img = argv[1];
    const char* filter = (argc >= 3) ? argv[2] : nullptr;

    CuDriverSession session;
    bool all_ok = true;
    int run = 0, passed = 0;
    for (const auto& tc : kCases) {
        if (filter && std::strcmp(filter, tc.name) != 0) continue;
        ++run;
        if (run_one(session, device_img, tc, std::cerr)) {
            ++passed;
        } else {
            all_ok = false;
            break;  // stop at first fail to keep diagnostics clear
        }
    }
    if (run == 0) {
        std::cerr << "no matching kernel for filter " << (filter ? filter : "<none>") << "\n";
        return 1;
    }
    if (all_ok) {
        std::cout << "TEST PASSED (" << passed << "/" << run << " kernels)\n";
        return 0;
    }
    return 1;
}
