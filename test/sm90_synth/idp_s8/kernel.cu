#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

extern "C" __global__ void kern_idp_s8(const int* __restrict__ a,
                                        const int* __restrict__ b,
                                        int* __restrict__ out, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    // __dp4a: dot product of 4 packed int8 pairs + int32 accumulator
    out[tid] = __dp4a(a[tid], b[tid], 0);
}

int main() {
    constexpr int N = 256;
    int h_a[N], h_b[N], h_out[N];
    for (int i = 0; i < N; i++) {
        int8_t a0 = (i % 7) - 3, a1 = (i % 5) - 2, a2 = (i % 3) - 1, a3 = (i % 11) - 5;
        int8_t b0 = (i % 13) - 6, b1 = (i % 9) - 4, b2 = (i % 6) - 3, b3 = (i % 4) - 2;
        h_a[i] = (a0 & 0xFF) | ((a1 & 0xFF) << 8) | ((a2 & 0xFF) << 16) | ((a3 & 0xFF) << 24);
        h_b[i] = (b0 & 0xFF) | ((b1 & 0xFF) << 8) | ((b2 & 0xFF) << 16) | ((b3 & 0xFF) << 24);
    }
    int *d_a, *d_b, *d_out;
    cudaMalloc(&d_a, N * sizeof(int));
    cudaMalloc(&d_b, N * sizeof(int));
    cudaMalloc(&d_out, N * sizeof(int));
    cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(int), cudaMemcpyHostToDevice);
    kern_idp_s8<<<(N + 255) / 256, 256>>>(d_a, d_b, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);
    FILE* f = fopen("ref_output.bin", "wb");
    fwrite(h_out, sizeof(int), N, f); fclose(f);
    int8_t a0 = -3, a1 = -2, a2 = -1, a3 = -5;
    int8_t b0 = -6, b1 = -4, b2 = -3, b3 = -2;
    int expected = a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3;
    bool ok = (h_out[0] == expected);
    printf(ok ? "TEST PASSED\n" : "TEST FAILED (got %d exp %d)\n", h_out[0], expected);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_out);
    return ok ? 0 : 1;
}
