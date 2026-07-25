#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

// Software FP16 <-> FP32 conversion (IEEE 754 half-precision)
static inline uint16_t fp32_to_fp16(float f) {
    uint32_t x; memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000;
    int exp = ((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x007FFFFF;
    if (((x >> 23) & 0xFF) == 0xFF) return (uint16_t)(sign | 0x7C00 | ((mant >> 13) ? 0x200 : 0));
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x00800000;
        int shift = 14 - exp;
        uint32_t half = 1u << (shift - 1);
        uint32_t dropped = mant & ((1u << shift) - 1);
        mant >>= shift;
        if (dropped > half || (dropped == half && (mant & 1)))
            mant++;
        return (uint16_t)(sign | mant);
    }
    if (exp >= 31) return (uint16_t)(sign | 0x7C00);
    uint32_t dropped = mant & 0x1FFF;
    mant >>= 13;
    if (dropped > 0x1000 || (dropped == 0x1000 && (mant & 1)))
        mant++;
    if (mant > 0x3FF) { mant = 0; exp++; }
    if (exp >= 31) return (uint16_t)(sign | 0x7C00);
    return (uint16_t)(sign | (exp << 10) | mant);
}
static inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t result;
    if (exp == 0) {
        if (mant == 0) { result = sign; }
        else { // subnormal
            exp = 1;
            while (!(mant & 0x400)) { mant <<= 1; exp--; }
            mant &= 0x3FF;
            result = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        result = sign | 0x7F800000 | (mant << 13);
    } else {
        result = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float f; memcpy(&f, &result, 4);
    return f;
}


extern "C" void F3e_f2f_roundtrip();

int main() {
    constexpr int N = 1024;
    float *h_in = new float[N], *h_out = new float[N];
    memset(h_out, 0, N*sizeof(float));
    float vals[] = {0.1f, 0.3333f, 1.5f, 65504.0f, 0.000061f, -0.5f, 100.25f, 3.14159f};
    for (int i = 0; i < N; i++)
        h_in[i] = vals[i%8] * ((i/8)%2==0 ? 1.0f : -1.0f);
    launchKernel(F3e_f2f_roundtrip, (N+255)/256, 256, h_in, h_out, N);
    bool ok = true;
    for (int i = 0; i < N; i++) {
        float expected = fp16_to_fp32(fp32_to_fp16(h_in[i]));
        if (fabsf(h_out[i] - expected) > 1e-6f) {
            std::cout << "FAIL at " << i << ": got " << h_out[i] << " exp " << expected << std::endl;
            ok = false; break;
        }
    }
    if (ok) std::cout << "TEST PASSED" << std::endl;
    delete[] h_in; delete[] h_out;
    return ok ? 0 : 1;
}
