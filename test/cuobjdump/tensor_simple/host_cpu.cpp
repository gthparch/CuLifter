#include <cstdint>
#include <cstring>
#include <iostream>
#include <cmath>
#include <random>
#include <vector>
#include "kernel_wrapper.h"

// 16x16x16 single-WMMA matmul: C = A * B
extern "C" void wmma_simple() asm("_Z11wmma_simpleP6__halfS0_Pf");

static uint16_t f32_to_half(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, 4);
    uint32_t sign = (bits >> 31) & 0x1;
    int32_t exp = ((bits >> 23) & 0xff) - 127 + 15;
    uint32_t frac = (bits >> 13) & 0x3ff;
    if (exp <= 0) return (uint16_t)(sign << 15);
    if (exp >= 31) return (uint16_t)((sign << 15) | (0x1f << 10));
    return (uint16_t)((sign << 15) | (exp << 10) | frac);
}

static float half_to_f32(uint16_t h) {
    uint32_t sign = (h >> 15) & 0x1;
    int32_t exp = (h >> 10) & 0x1f;
    uint32_t frac = h & 0x3ff;
    uint32_t bits;
    if (exp == 0) bits = sign << 31;
    else if (exp == 31) bits = (sign << 31) | (0xff << 23);
    else bits = (sign << 31) | ((exp - 15 + 127) << 23) | (frac << 13);
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

int main() {
    constexpr int M = 16, N = 16, K = 16;
    std::vector<uint16_t> a(M*K), b(K*N);
    std::vector<float>    c(M*N, 0.0f), ref(M*N, 0.0f);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (auto& x : a) x = f32_to_half(dist(gen));
    for (auto& x : b) x = f32_to_half(dist(gen));

    launchKernel(wmma_simple, 1, 32, a.data(), b.data(), c.data());

    for (int i = 0; i < M; ++i) for (int j = 0; j < N; ++j) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k)
            acc += half_to_f32(a[i*K+k]) * half_to_f32(b[k*N+j]);
        ref[i*N+j] = acc;
    }
    bool ok = true;
    for (size_t idx = 0; idx < c.size(); ++idx) {
        if (std::fabs(c[idx] - ref[idx]) > 1e-1f) {
            std::cerr << "FAIL at " << idx << ": got " << c[idx]
                      << ", expected " << ref[idx] << "\n";
            ok = false;
            break;
        }
    }
    if (ok) std::cout << "TEST PASSED\n";
    return ok ? 0 : 1;
}
