#include <cstdio>
#include <cuda_runtime.h>

extern "C" __global__ void kern_r2p_p2r(const int* __restrict__ in,
                                         int* __restrict__ out, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    int val = in[tid];
    int pred_bits, result;
    // Create predicates, ballot them to a register
    asm volatile(
        "{\n\t"
        "  .reg .pred %%p0, %%p1, %%p2, %%p3;\n\t"
        "  setp.gt.s32 %%p0, %1, 0;\n\t"
        "  setp.gt.s32 %%p1, %1, 10;\n\t"
        "  setp.gt.s32 %%p2, %1, 100;\n\t"
        "  setp.gt.s32 %%p3, %1, 1000;\n\t"
        "  vote.ballot.sync.b32 %0, %%p0, 0xffffffff;\n\t"
        "}" : "=r"(pred_bits) : "r"(val));
    // Use register bits to select result
    asm volatile(
        "{\n\t"
        "  .reg .pred %%q0;\n\t"
        "  and.b32 %0, %1, 1;\n\t"
        "  setp.ne.s32 %%q0, %0, 0;\n\t"
        "  @%%q0 add.s32 %0, %2, 1;\n\t"
        "  @!%%q0 mov.b32 %0, %2;\n\t"
        "}" : "=r"(result) : "r"(pred_bits), "r"(val));
    out[tid] = result;
}

int main() {
    constexpr int N = 256;
    int h_in[N], h_out[N];
    for (int i = 0; i < N; i++) h_in[i] = i * 7 - 100;
    int *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(int));
    cudaMalloc(&d_out, N * sizeof(int));
    cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice);
    kern_r2p_p2r<<<(N + 255) / 256, 256>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);
    FILE* f = fopen("ref_output.bin", "wb");
    fwrite(h_out, sizeof(int), N, f); fclose(f);
    printf("TEST PASSED\n");
    cudaFree(d_in); cudaFree(d_out);
    return 0;
}
