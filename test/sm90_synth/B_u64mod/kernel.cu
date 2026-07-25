// Microbench: 64-bit multiply + modulo by a runtime 32-bit divisor.
// Mirrors the core of mr-cuda's modular_exponentiation32 ((uint64)d*A % n),
// which the lifted mr32_sf/mr32_eff get wrong for every input.
#include <cstdint>

extern "C" __global__ void kern_u64mod(uint32_t* out, const uint32_t* a,
                                       const uint32_t* b, const uint32_t* n,
                                       int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    out[i] = (uint32_t)(((uint64_t)a[i] * b[i]) % n[i]);
}
