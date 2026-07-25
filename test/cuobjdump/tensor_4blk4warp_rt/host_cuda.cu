#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include <cuda_fp16.h>

#include "cuda_driver_helpers.h"

namespace { constexpr char kKernelName[] = "wmma_4blk4warp_rt"; }

int main(int argc, char** argv) {
    if (argc < 2) { std::cerr << "usage: " << argv[0] << " <cubin|ptx>\n"; return 1; }
    int M = 64, N = 64, K = 16;

    std::vector<__half> a(M*K), b(K*N);
    std::vector<float>  c(M*N, 0.0f), ref(M*N, 0.0f);

    std::mt19937 gen(12345);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (auto& x : a) x = __float2half(dist(gen));
    for (auto& x : b) x = __float2half(dist(gen));

    CuDriverSession session;
    CUfunction fn = session.loadKernel(argv[1], kKernelName);

    CUdeviceptr d_a{}, d_b{}, d_c{};
    CUCHK(cuMemAlloc(&d_a, a.size()*sizeof(__half)));
    CUCHK(cuMemAlloc(&d_b, b.size()*sizeof(__half)));
    CUCHK(cuMemAlloc(&d_c, c.size()*sizeof(float)));
    CUCHK(cuMemcpyHtoD(d_a, a.data(), a.size()*sizeof(__half)));
    CUCHK(cuMemcpyHtoD(d_b, b.data(), b.size()*sizeof(__half)));
    CUCHK(cuMemsetD8(d_c, 0, c.size()*sizeof(float)));

    void* params[] = {&d_a, &d_b, &d_c, &M, &N, &K};
    CUCHK(cuLaunchKernel(fn, 4, 1, 1, 128, 1, 1, 0, nullptr, params, nullptr));
    CUCHK(cuCtxSynchronize());
    CUCHK(cuMemcpyDtoH(c.data(), d_c, c.size()*sizeof(float)));

    bool ok = true;
    for (int i = 0; i < M; ++i) for (int j = 0; j < N; ++j) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k)
            acc += __half2float(a[i*K+k]) * __half2float(b[k*N+j]);
        ref[i*N+j] = acc;
    }
    for (size_t idx = 0; idx < c.size(); ++idx) {
        if (std::fabs(c[idx]-ref[idx]) > 1e-1f) {
            std::cerr << "FAIL at " << idx << ": got " << c[idx] << ", expected " << ref[idx] << "\n";
            ok = false; break;
        }
    }
    if (ok) std::cout << "TEST PASSED\n";
    CUCHK(cuMemFree(d_a)); CUCHK(cuMemFree(d_b)); CUCHK(cuMemFree(d_c));
    return ok ? 0 : 1;
}
