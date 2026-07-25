#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

extern "C" __global__ void SH17_smem_type_pun(const float* in, unsigned int* out, int N) {
    extern __shared__ char smem_raw[];
    float* sf = (float*)smem_raw;
    unsigned int* si = (unsigned int*)smem_raw;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) sf[threadIdx.x] = in[i];
    __syncthreads();
    if (i < N) out[i] = si[threadIdx.x];
}

int main(int argc, char** argv) {
    constexpr int N = 1024, BS = 256;
    float h_in[N]; unsigned int h_out[N];
    for (int i = 0; i < N; i++) h_in[i] = (float)(i*3+7);
    memset(h_out, 0, sizeof(h_out));
    float *d_in; unsigned int *d_out;
    cudaMalloc(&d_in, N*sizeof(float)); cudaMalloc(&d_out, N*sizeof(unsigned int));
    cudaMemcpy(d_in, h_in, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, N*sizeof(unsigned int));
    SH17_smem_type_pun<<<(N+BS-1)/BS, BS, BS*sizeof(float)>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N*sizeof(unsigned int), cudaMemcpyDeviceToHost);
    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb"); fwrite(h_out, sizeof(unsigned int), N, f); fclose(f);
    }
    bool ok = true;
    for (int i = 0; i < N; i++) {
        unsigned int expected; memcpy(&expected, &h_in[i], 4);
        if (h_out[i] != expected) { printf("FAIL at %d\n",i); ok=false; break; }
    }
    printf(ok ? "TEST PASSED\n" : "FAIL\n");
    cudaFree(d_in); cudaFree(d_out);
    return ok ? 0 : 1;
}
