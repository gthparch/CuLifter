#include <cstdio>
#include <cuda_runtime.h>

__device__ unsigned int magic_table[8] = {
    0x1111aaaau, 0x2222bbbbu, 0x3333ccccu, 0x4444ddddu,
    0x5555eeeeu, 0x6666ffffu, 0x77770123u, 0x88884567u,
};
__device__ unsigned int zero_table[4];  // NOBITS -> .nv.global

extern "C" __global__ void kern_nvglobal_table(unsigned int* out,
                                               const int* idx, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    out[tid] = magic_table[idx[tid] & 7] + zero_table[idx[tid] & 3];
}

int main() {
    constexpr int N = 256;
    unsigned int ho[N];
    int hi[N];
    for (int i = 0; i < N; i++) hi[i] = i * 3 + 1;
    unsigned int* dout; int* didx;
    cudaMalloc(&dout, N * 4);
    cudaMalloc(&didx, N * 4);
    cudaMemcpy(didx, hi, N * 4, cudaMemcpyHostToDevice);
    kern_nvglobal_table<<<1, 256>>>(dout, didx, N);
    cudaDeviceSynchronize();
    cudaMemcpy(ho, dout, N * 4, cudaMemcpyDeviceToHost);
    FILE* f = fopen("ref_output.bin", "wb");
    fwrite(ho, 4, N, f);
    fclose(f);
    // idx[0]=1 -> magic_table[1]; idx[1]=4 -> magic_table[4]; zero_table adds 0
    bool ok = ho[0] == 0x2222bbbbu && ho[1] == 0x5555eeeeu;
    printf(ok ? "TEST PASSED\n" : "TEST FAILED\n");
    cudaFree(dout); cudaFree(didx);
    return ok ? 0 : 1;
}
