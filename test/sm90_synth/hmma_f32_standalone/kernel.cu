// HMMA.16816.F32 standalone test: single mma.sync.aligned.m16n8k16 operation
// Uses inline PTX to emit exactly one mma.sync instruction.
// The kernel sets up known FP16 values in A and B fragments,
// runs HMMA with C=0, and stores the result D.

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Helper to pack two FP16 values into a uint32
static __host__ __device__ uint32_t pack_fp16(half a, half b) {
    uint32_t lo = __half_as_ushort(a);
    uint32_t hi = __half_as_ushort(b);
    return lo | (hi << 16);
}

extern "C" __global__ void kern_hmma_f32_standalone(
    const uint32_t* __restrict__ a_packed,  // 4 packed fp16 pairs per thread
    const uint32_t* __restrict__ b_packed,  // 2 packed fp16 pairs per thread
    float* __restrict__ d_out,              // 4 f32 outputs per thread
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

    // C accumulator = 0
    float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;
    float d0, d1, d2, d3;

    // mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3)
    );

    // Store results
    d_out[gid * 4 + 0] = d0;
    d_out[gid * 4 + 1] = d1;
    d_out[gid * 4 + 2] = d2;
    d_out[gid * 4 + 3] = d3;
}

int main() {
    constexpr int WARP_SIZE = 32;
    constexpr int N = WARP_SIZE;

    // Set up known A and B fragments for all threads
    // Each thread gets 4 packed A regs + 2 packed B regs
    uint32_t h_a[N * 4], h_b[N * 2];
    float h_d[N * 4];

    // Initialize with simple values: A elements = 1.0h, B elements = 1.0h
    half one = __float2half(1.0f);
    half two = __float2half(2.0f);
    uint32_t pack_ones = pack_fp16(one, one);
    uint32_t pack_twos = pack_fp16(two, two);

    for (int i = 0; i < N * 4; i++) h_a[i] = pack_ones;  // A = all 1.0
    for (int i = 0; i < N * 2; i++) h_b[i] = pack_twos;  // B = all 2.0

    memset(h_d, 0, sizeof(h_d));

    uint32_t *d_a, *d_b;
    float *d_d;
    cudaMalloc(&d_a, N * 4 * sizeof(uint32_t));
    cudaMalloc(&d_b, N * 2 * sizeof(uint32_t));
    cudaMalloc(&d_d, N * 4 * sizeof(float));
    cudaMemcpy(d_a, h_a, N * 4 * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * 2 * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemset(d_d, 0, N * 4 * sizeof(float));

    kern_hmma_f32_standalone<<<1, WARP_SIZE>>>(d_a, d_b, d_d, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_d, d_d, N * 4 * sizeof(float), cudaMemcpyDeviceToHost);

    // Save reference output
    FILE* f = fopen("ref_output.bin", "wb");
    fwrite(h_d, sizeof(float), N * 4, f);
    fclose(f);

    // Print thread 0's results for debugging
    printf("Thread 0 D: %f %f %f %f\n", h_d[0], h_d[1], h_d[2], h_d[3]);

    // Verify: with A=1.0, B=2.0, C=0, the result should be non-zero
    bool ok = true;
    for (int i = 0; i < N * 4; i++) {
        if (std::isnan(h_d[i])) { printf("NaN at %d\n", i); ok = false; break; }
    }
    // Check thread 0's d0 is non-zero (should be sum of products)
    if (h_d[0] == 0.0f) { printf("d0 is zero, expected non-zero\n"); ok = false; }

    printf(ok ? "TEST PASSED\n" : "TEST FAILED\n");

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_d);
    return ok ? 0 : 1;
}
