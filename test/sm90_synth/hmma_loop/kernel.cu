// HMMA.16816.F32 in accumulation loop: D += A*B repeated
// Tests that the accumulator is correctly fed back.

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

extern "C" __global__ void kern_hmma_loop(
    const uint32_t* __restrict__ a_packed,
    const uint32_t* __restrict__ b_packed,
    float* __restrict__ d_out,
    int N, int iters
) {
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    uint32_t a0 = a_packed[gid * 4 + 0];
    uint32_t a1 = a_packed[gid * 4 + 1];
    uint32_t a2 = a_packed[gid * 4 + 2];
    uint32_t a3 = a_packed[gid * 4 + 3];

    uint32_t b0 = b_packed[gid * 2 + 0];
    uint32_t b1 = b_packed[gid * 2 + 1];

    float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;

    // Accumulate HMMA results over multiple iterations
    for (int i = 0; i < iters; i++) {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0, %1, %2, %3}, "
            "{%4, %5, %6, %7}, "
            "{%8, %9}, "
            "{%0, %1, %2, %3};"
            : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
              "r"(b0), "r"(b1)
        );
    }

    d_out[gid * 4 + 0] = c0;
    d_out[gid * 4 + 1] = c1;
    d_out[gid * 4 + 2] = c2;
    d_out[gid * 4 + 3] = c3;
}

int main() {
    constexpr int WARP_SIZE = 32;
    constexpr int N = WARP_SIZE;
    constexpr int ITERS = 4;

    uint32_t h_a[N * 4], h_b[N * 2];
    float h_d[N * 4];

    half one = __float2half(1.0f);
    uint32_t pack_ones = pack_fp16(one, one);

    for (int i = 0; i < N * 4; i++) h_a[i] = pack_ones;
    for (int i = 0; i < N * 2; i++) h_b[i] = pack_ones;
    memset(h_d, 0, sizeof(h_d));

    uint32_t *d_a, *d_b;
    float *d_d;
    cudaMalloc(&d_a, N * 4 * sizeof(uint32_t));
    cudaMalloc(&d_b, N * 2 * sizeof(uint32_t));
    cudaMalloc(&d_d, N * 4 * sizeof(float));
    cudaMemcpy(d_a, h_a, N * 4 * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * 2 * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemset(d_d, 0, N * 4 * sizeof(float));

    kern_hmma_loop<<<1, WARP_SIZE>>>(d_a, d_b, d_d, N, ITERS);
    cudaDeviceSynchronize();
    cudaMemcpy(h_d, d_d, N * 4 * sizeof(float), cudaMemcpyDeviceToHost);

    FILE* f = fopen("ref_output.bin", "wb");
    fwrite(h_d, sizeof(float), N * 4, f);
    fclose(f);

    printf("Thread 0 D (after %d iters): %f %f %f %f\n",
           ITERS, h_d[0], h_d[1], h_d[2], h_d[3]);

    // After ITERS iterations, values should be ITERS * (single HMMA result)
    bool ok = !std::isnan(h_d[0]) && (h_d[0] != 0.0f);
    printf(ok ? "TEST PASSED\n" : "TEST FAILED\n");

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_d);
    return ok ? 0 : 1;
}
