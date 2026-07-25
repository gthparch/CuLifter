#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

extern "C" __global__ void SHa_two_arrays(const float* a, const float* b, float* out, int N) {
    __shared__ float sa[256], sb[256];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) { sa[threadIdx.x] = a[i]; sb[threadIdx.x] = b[i]; }
    __syncthreads();
    if (i < N) out[i] = sa[threadIdx.x] + sb[threadIdx.x];
}

int main(int argc, char** argv) {
    constexpr int N = 1024, BS = 256;
    float h_a[N], h_b[N], h_out[N];
    for (int i = 0; i < N; i++) { h_a[i]=(float)(i%7+1); h_b[i]=(float)(i%5+1); }
    memset(h_out, 0, sizeof(h_out));
    float *d_a, *d_b, *d_out;
    cudaMalloc(&d_a, N*sizeof(float)); cudaMalloc(&d_b, N*sizeof(float));
    cudaMalloc(&d_out, N*sizeof(float));
    cudaMemcpy(d_a, h_a, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, N*sizeof(float));
    SHa_two_arrays<<<(N+BS-1)/BS, BS>>>(d_a, d_b, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N*sizeof(float), cudaMemcpyDeviceToHost);
    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb"); fwrite(h_out, sizeof(float), N, f); fclose(f);
    }
    bool ok = true;
    for (int i = 0; i < N; i++) {
        float exp = h_a[i]+h_b[i];
        if (fabsf(h_out[i]-exp)>1e-5f) { printf("FAIL at %d\n",i); ok=false; break; }
    }
    printf(ok ? "TEST PASSED\n" : "FAIL\n");
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_out);
    return ok ? 0 : 1;
}
