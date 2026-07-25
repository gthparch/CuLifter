#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

// Force ULDC.64 by reading a 64-bit kernel parameter (pointer)
// through uniform register path
extern "C" __global__ void kern_uldc64(const float* __restrict__ in,
                                        float* __restrict__ out,
                                        long long N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    // The pointer args and N (64-bit) generate ULDC.64 for const bank reads
    out[tid] = in[tid] * 2.0f + (float)(N & 0xFF);
}

int main() {
    constexpr int N = 256;
    float h_in[N], h_out[N];
    for (int i = 0; i < N; i++) h_in[i] = (float)(i % 17 + 1) * 0.1f;
    float *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);
    kern_uldc64<<<(N + 255) / 256, 256>>>(d_in, d_out, (long long)N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);
    FILE* f = fopen("ref_output.bin", "wb");
    fwrite(h_out, sizeof(float), N, f); fclose(f);
    bool ok = true;
    for (int i = 0; i < N; i++) {
        float exp = h_in[i] * 2.0f + (float)(N & 0xFF);
        if (fabsf(h_out[i] - exp) > 1e-3f) { ok = false; break; }
    }
    printf(ok ? "TEST PASSED\n" : "TEST FAILED\n");
    cudaFree(d_in); cudaFree(d_out);
    return ok ? 0 : 1;
}
