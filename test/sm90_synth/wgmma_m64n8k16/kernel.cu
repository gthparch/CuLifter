// HGMMA/WGMMA M64N8K16.F32 test
// Tests HGMMA.64x8x16.F32 register-based WGMMA form.
// The SASS kernel.sass uses HGMMA.64x8x16.F32 (hand-crafted from HMMA.16816.F32).
// On x86, HGMMA is decomposed to scalar FMA identical to HMMA.16816.F32 (warp 0, lane 0).
//
// Note: This kernel.cu was written for reference but does not compile due to WGMMA
// inline PTX constraints. The SASS was hand-crafted instead.

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

static __host__ __device__ uint32_t pack_fp16(half a, half b) {
    uint32_t lo = __half_as_ushort(a);
    uint32_t hi = __half_as_ushort(b);
    return lo | (hi << 16);
}

extern "C" __global__ void kern_wgmma_m64n8k16(
    const uint32_t* __restrict__ a_packed,
    const uint32_t* __restrict__ b_packed,
    float* __restrict__ d_out,
    int N
) {
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    // Load A fragment (4 registers)
    uint32_t a0 = a_packed[gid * 4 + 0];
    uint32_t a1 = a_packed[gid * 4 + 1];
    uint32_t a2 = a_packed[gid * 4 + 2];
    uint32_t a3 = a_packed[gid * 4 + 3];

    // Load B fragment (2 registers)
    uint32_t b0 = b_packed[gid * 2 + 0];
    uint32_t b1 = b_packed[gid * 2 + 1];

    // D/C accumulator = 0
    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;

    // wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16
    // Register-register form (operand A and B from registers)
    // scaleD=1, scaleA=1, scaleB=1, transB=0
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "1, 1, 1, 0, 0;\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1)
    );

    // Store results
    d_out[gid * 4 + 0] = d0;
    d_out[gid * 4 + 1] = d1;
    d_out[gid * 4 + 2] = d2;
    d_out[gid * 4 + 3] = d3;
}

int main() {
    constexpr int WARPGROUP = 128;  // 4 warps = 1 warpgroup
    constexpr int N = WARPGROUP;

    uint32_t h_a[N * 4], h_b[N * 2];
    float h_d[N * 4];

    half one = __float2half(1.0f);
    half two = __float2half(2.0f);
    uint32_t pack_ones = pack_fp16(one, one);
    uint32_t pack_twos = pack_fp16(two, two);

    for (int i = 0; i < N * 4; i++) h_a[i] = pack_ones;
    for (int i = 0; i < N * 2; i++) h_b[i] = pack_twos;
    memset(h_d, 0, sizeof(h_d));

    uint32_t *d_a, *d_b;
    float *d_d;
    cudaMalloc(&d_a, N * 4 * sizeof(uint32_t));
    cudaMalloc(&d_b, N * 2 * sizeof(uint32_t));
    cudaMalloc(&d_d, N * 4 * sizeof(float));
    cudaMemcpy(d_a, h_a, N * 4 * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * 2 * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemset(d_d, 0, N * 4 * sizeof(float));

    kern_wgmma_m64n8k16<<<1, WARPGROUP>>>(d_a, d_b, d_d, N);
    cudaDeviceSynchronize();

    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaMemcpy(h_d, d_d, N * 4 * sizeof(float), cudaMemcpyDeviceToHost);

    FILE* f = fopen("ref_output.bin", "wb");
    fwrite(h_d, sizeof(float), N * 4, f);
    fclose(f);

    printf("Thread 0 D: %f %f %f %f\n", h_d[0], h_d[1], h_d[2], h_d[3]);

    bool ok = true;
    for (int i = 0; i < N * 4; i++) {
        if (std::isnan(h_d[i])) { printf("NaN at %d\n", i); ok = false; break; }
    }

    printf(ok ? "TEST PASSED\n" : "TEST FAILED\n");

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_d);
    return ok ? 0 : 1;
}
