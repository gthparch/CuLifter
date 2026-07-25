#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

extern "C" __global__ void SHa_dynamic_int(const int* in, int* out, int N) {
    extern __shared__ int sd[];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) sd[threadIdx.x] = in[i];
    __syncthreads();
    if (i < N) out[i] = sd[threadIdx.x];
}

int main(int argc, char** argv) {
    constexpr int N = 1024, BS = 256;
    int h_in[N], h_out[N];
    for (int i = 0; i < N; i++) h_in[i] = i*3+7;
    memset(h_out, 0, sizeof(h_out));
    int *d_in, *d_out;
    cudaMalloc(&d_in, N*sizeof(int)); cudaMalloc(&d_out, N*sizeof(int));
    cudaMemcpy(d_in, h_in, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, N*sizeof(int));
    SHa_dynamic_int<<<(N+BS-1)/BS, BS, BS*sizeof(int)>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N*sizeof(int), cudaMemcpyDeviceToHost);
    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb"); fwrite(h_out, sizeof(int), N, f); fclose(f);
    }
    bool ok = true;
    for (int i = 0; i < N; i++)
        if (h_out[i] != h_in[i]) { printf("FAIL at %d\n",i); ok=false; break; }
    printf(ok ? "TEST PASSED\n" : "FAIL\n");
    cudaFree(d_in); cudaFree(d_out);
    return ok ? 0 : 1;
}
