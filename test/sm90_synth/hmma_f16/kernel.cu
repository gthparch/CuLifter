// HMMA.16816.F16 test: mma.sync.aligned.m16n8k16 with FP16 accumulation
// Uses inline PTX to emit mma.sync with .f16 accumulator type.
// For F16 accumulation: C/D have 2 registers (each v2f16), not 4.

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

extern "C" __global__ void kern_hmma_f16(
    const uint32_t* __restrict__ a_packed,
    const uint32_t* __restrict__ b_packed,
    uint32_t* __restrict__ d_out,
    int N
) {
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    uint32_t a0 = a_packed[gid * 4 + 0];
    uint32_t a1 = a_packed[gid * 4 + 1];
    uint32_t a2 = a_packed[gid * 4 + 2];
    uint32_t a3 = a_packed[gid * 4 + 3];

    uint32_t b0 = b_packed[gid * 2 + 0];
    uint32_t b1 = b_packed[gid * 2 + 1];

    // C accumulator = 0 (packed FP16 zeros); F16 accum uses 2 regs, not 4
    uint32_t c0 = 0, c1 = 0;
    uint32_t d0, d1;

    // mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16
    // D/C: 2 registers (each v2f16 = 2 packed half)
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0, %1}, "
        "{%2, %3, %4, %5}, "
        "{%6, %7}, "
        "{%8, %9};"
        : "=r"(d0), "=r"(d1)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "r"(c0), "r"(c1)
    );

    d_out[gid * 2 + 0] = d0;
    d_out[gid * 2 + 1] = d1;
}

int main() {
    constexpr int WARP_SIZE = 32;
    constexpr int N = WARP_SIZE;

    uint32_t h_a[N * 4], h_b[N * 2];
    uint32_t h_d[N * 2];

    half one = __float2half(1.0f);
    half two = __float2half(2.0f);
    uint32_t pack_ones = pack_fp16(one, one);
    uint32_t pack_twos = pack_fp16(two, two);

    for (int i = 0; i < N * 4; i++) h_a[i] = pack_ones;
    for (int i = 0; i < N * 2; i++) h_b[i] = pack_twos;
    memset(h_d, 0, sizeof(h_d));

    uint32_t *d_a, *d_b, *d_d;
    cudaMalloc(&d_a, N * 4 * sizeof(uint32_t));
    cudaMalloc(&d_b, N * 2 * sizeof(uint32_t));
    cudaMalloc(&d_d, N * 2 * sizeof(uint32_t));
    cudaMemcpy(d_a, h_a, N * 4 * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * 2 * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemset(d_d, 0, N * 2 * sizeof(uint32_t));

    kern_hmma_f16<<<1, WARP_SIZE>>>(d_a, d_b, d_d, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_d, d_d, N * 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    FILE* f = fopen("ref_output.bin", "wb");
    fwrite(h_d, sizeof(uint32_t), N * 2, f);
    fclose(f);

    // Print thread 0 results (unpacked)
    half d0_lo = __ushort_as_half(h_d[0] & 0xFFFF);
    half d0_hi = __ushort_as_half((h_d[0] >> 16) & 0xFFFF);
    printf("Thread 0 D[0]: lo=%f hi=%f\n", __half2float(d0_lo), __half2float(d0_hi));

    bool ok = (h_d[0] != 0);
    printf(ok ? "TEST PASSED\n" : "TEST FAILED\n");

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_d);
    return ok ? 0 : 1;
}
