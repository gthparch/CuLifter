#include <cstdio>
#include <cuda_runtime.h>

extern "C" __global__ void kern_elect(int* __restrict__ out, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    int is_elected;
    asm volatile(
        "{\n\t"
        "  .reg .pred %%p;\n\t"
        "  elect.sync _|%%p, 0xffffffff;\n\t"
        "  selp.b32 %0, 1, 0, %%p;\n\t"
        "}" : "=r"(is_elected));
    out[tid] = is_elected;
}

int main() {
    constexpr int N = 256;
    int h_out[N];
    int *d_out;
    cudaMalloc(&d_out, N * sizeof(int));
    cudaMemset(d_out, 0, N * sizeof(int));
    kern_elect<<<(N + 255) / 256, 256>>>(d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);
    FILE* f = fopen("ref_output.bin", "wb");
    fwrite(h_out, sizeof(int), N, f); fclose(f);
    // Thread 0 (lane 0 of warp 0) should be elected
    bool ok = (h_out[0] == 1);
    printf(ok ? "TEST PASSED\n" : "TEST FAILED\n");
    cudaFree(d_out);
    return ok ? 0 : 1;
}
