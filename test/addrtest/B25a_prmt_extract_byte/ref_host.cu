#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

extern "C" __global__ void B25a_prmt_extract_byte(const unsigned int* in,
                                   unsigned int* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        unsigned int x = in[i];
        unsigned int b0 = x & 0xFFu;
        unsigned int b1 = (x >> 8) & 0xFFu;
        unsigned int b2 = (x >> 16) & 0xFFu;
        unsigned int b3 = (x >> 24) & 0xFFu;
        out[i] = b0 + b1 + b2 + b3;
    }
}

int main(int argc, char** argv) {
    constexpr int N = 1024, BS = 256;
    unsigned int h_in[N], h_out[N];
    for (int i = 0; i < N; i++) h_in[i] = (unsigned int)(i * 0x01010101u + 0x03050709u);
    memset(h_out, 0, sizeof(h_out));
    unsigned int *d_in, *d_out;
    cudaMalloc(&d_in, N*sizeof(unsigned int)); cudaMalloc(&d_out, N*sizeof(unsigned int));
    cudaMemcpy(d_in, h_in, N*sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, N*sizeof(unsigned int));
    B25a_prmt_extract_byte<<<(N+BS-1)/BS, BS>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N*sizeof(unsigned int), cudaMemcpyDeviceToHost);
    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb"); fwrite(h_out, sizeof(unsigned int), N, f); fclose(f);
    }
    bool ok = true;
    for (int i = 0; i < N; i++) {
        unsigned int x = h_in[i];
        unsigned int exp = (x&0xFF)+((x>>8)&0xFF)+((x>>16)&0xFF)+((x>>24)&0xFF);
        if (h_out[i] != exp) { printf("FAIL at %d\n",i); ok=false; break; }
    }
    printf(ok ? "TEST PASSED\n" : "FAIL\n");
    cudaFree(d_in); cudaFree(d_out);
    return ok ? 0 : 1;
}
