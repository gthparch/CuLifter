#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

extern "C" __global__ void SHt_u8_copy(const unsigned char* in, unsigned char* out, int N) {
    __shared__ unsigned char s[256];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) s[threadIdx.x] = in[i];
    __syncthreads();
    if (i < N) out[i] = s[threadIdx.x];
}

int main(int argc, char** argv) {
    constexpr int N = 1024, BS = 256;
    unsigned char h_in[N], h_out[N];
    for (int i = 0; i < N; i++) h_in[i] = (unsigned char)(i%200);
    memset(h_out, 0, sizeof(h_out));
    unsigned char *d_in, *d_out;
    cudaMalloc(&d_in, N*sizeof(unsigned char)); cudaMalloc(&d_out, N*sizeof(unsigned char));
    cudaMemcpy(d_in, h_in, N*sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, N*sizeof(unsigned char));
    SHt_u8_copy<<<(N+BS-1)/BS, BS>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N*sizeof(unsigned char), cudaMemcpyDeviceToHost);
    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb"); fwrite(h_out, sizeof(unsigned char), N, f); fclose(f);
    }
    bool ok = true;
    for (int i = 0; i < N; i++)
        if (h_out[i]!=h_in[i]) { printf("FAIL at %d\n",i); ok=false; break; }
    printf(ok ? "TEST PASSED\n" : "FAIL\n");
    cudaFree(d_in); cudaFree(d_out);
    return ok ? 0 : 1;
}
