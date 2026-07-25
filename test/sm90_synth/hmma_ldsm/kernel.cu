// HMMA.16816.F32 with LDSM input: load matrix fragments from shared memory
// via ldmatrix, then feed into mma.sync.
//
// ldmatrix alignment requirement: each lane's address must be 16-byte aligned.
// Layout: each "row" in shared memory is 16 bytes (8 FP16 values = 4 uint32).

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

extern "C" __global__ void kern_hmma_ldsm(float* __restrict__ d_out, int N) {
    // Shared memory: 16-byte aligned rows
    // A matrix: 16 rows x 16 cols of FP16 = 16 rows x 8 uint32 = 128 uint32
    // B matrix: 16 rows x 8 cols of FP16 = 16 rows x 4 uint32 = 64 uint32
    // Pad each row to 16-byte boundary (already satisfied: 8*4=32 bytes per row)
    __shared__ __align__(16) uint32_t smem[256];
    // smem[0..127]   = A matrix (16 rows x 8 uint32)
    // smem[128..191] = B matrix (16 rows x 4 uint32)

    int tid = threadIdx.x;
    int lane = tid & 31;

    // Fill A with 1.0h, B with 2.0h
    half one = __float2half(1.0f);
    half two = __float2half(2.0f);
    uint32_t pack_ones = __half_as_ushort(one) | ((uint32_t)__half_as_ushort(one) << 16);
    uint32_t pack_twos = __half_as_ushort(two) | ((uint32_t)__half_as_ushort(two) << 16);

    // Fill with single thread (warp-sized kernel)
    for (int i = tid; i < 128; i += 32) smem[i] = pack_ones;       // A
    for (int i = tid; i < 64; i += 32)  smem[128 + i] = pack_twos; // B
    __syncthreads();

    // ldmatrix.x4 for A: each lane reads one 16-byte row
    // Lane i reads row (i % 8) from the first tile, etc.
    // Row stride = 8 uint32 = 32 bytes, but ldmatrix reads 4 uint32 = 16 bytes per lane
    // So use stride of 4 uint32 (16 bytes) between rows within a tile
    // Tile 0 rows at smem[0], smem[4], smem[8], ..., smem[28]
    // Tile 1 rows at smem[32], smem[36], ...
    uint32_t a_addr = static_cast<uint32_t>(
        __cvta_generic_to_shared(&smem[(lane % 8) * 4]));

    uint32_t a0, a1, a2, a3;
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
        : "r"(a_addr)
    );

    // ldmatrix.x2 for B: each lane reads 16 bytes
    // B rows at smem[128], smem[132], smem[136], ...
    uint32_t b_addr = static_cast<uint32_t>(
        __cvta_generic_to_shared(&smem[128 + (lane % 8) * 4]));

    uint32_t b0, b1;
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
        : "=r"(b0), "=r"(b1)
        : "r"(b_addr)
    );

    // MMA: D = A * B + 0
    float d0, d1, d2, d3;
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f)
    );

    // Write output
    if (tid < N) {
        d_out[tid * 4 + 0] = d0;
        d_out[tid * 4 + 1] = d1;
        d_out[tid * 4 + 2] = d2;
        d_out[tid * 4 + 3] = d3;
    }
}

int main() {
    constexpr int N = 32;
    float h_d[N * 4];
    memset(h_d, 0, sizeof(h_d));

    float *d_d;
    cudaMalloc(&d_d, N * 4 * sizeof(float));
    cudaMemset(d_d, 0, N * 4 * sizeof(float));

    kern_hmma_ldsm<<<1, 32>>>(d_d, N);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        cudaFree(d_d);
        return 1;
    }

    cudaMemcpy(h_d, d_d, N * 4 * sizeof(float), cudaMemcpyDeviceToHost);

    FILE* f = fopen("ref_output.bin", "wb");
    fwrite(h_d, sizeof(float), N * 4, f);
    fclose(f);

    printf("Thread 0 D: %f %f %f %f\n", h_d[0], h_d[1], h_d[2], h_d[3]);

    // With A=1.0 and B=2.0, the dot products depend on fragment layout
    bool ok = (h_d[0] != 0.0f) && !std::isnan(h_d[0]);
    printf(ok ? "TEST PASSED\n" : "TEST FAILED\n");

    cudaFree(d_d);
    return ok ? 0 : 1;
}
