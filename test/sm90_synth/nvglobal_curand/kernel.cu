#include <curand_kernel.h>

// XORWOW skip-ahead (subsequence=tid) reads precalc_xorwow_matrix from
// .nv.global.init via R_CUDA_ABS32_LO/HI_32 UMOV immediate relocs.
// The host curand XORWOW generator interleaves 4096 subsequences
// (element j = subsequence j's first draw), so thread tid's draw equals
// host generateUniform element tid bit-exactly.
extern "C" __global__ void kern_curand_xorwow(float* out, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    curandStateXORWOW st;
    curand_init(1234ULL, tid, 0, &st);
    out[tid] = curand_uniform(&st);
}
