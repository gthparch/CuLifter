#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

extern "C" __global__ void B26d_lop3_mux(const unsigned int* sel,
                                   const unsigned int* a,
                                   const unsigned int* b,
                                   unsigned int* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        out[i] = (sel[i] & a[i]) | (~sel[i] & b[i]);
    }
}

int main(int argc, char** argv) {
    constexpr int N = 1024, BS = 256;
    unsigned int h_sel[N], h_a[N], h_b[N], h_out[N];
    for (int i = 0; i < N; i++) {
        h_sel[i] = (unsigned int)(i*0xF0F0F0F0u+0x0F0F0F0Fu);
        h_a[i] = (unsigned int)(i*0x12345678u+1);
        h_b[i] = (unsigned int)(i*0x87654321u+2);
    }
    memset(h_out, 0, sizeof(h_out));
    unsigned int *d_sel, *d_a, *d_b, *d_out;
    cudaMalloc(&d_sel, N*sizeof(unsigned int)); cudaMalloc(&d_a, N*sizeof(unsigned int));
    cudaMalloc(&d_b, N*sizeof(unsigned int)); cudaMalloc(&d_out, N*sizeof(unsigned int));
    cudaMemcpy(d_sel, h_sel, N*sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a, h_a, N*sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N*sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, N*sizeof(unsigned int));
    B26d_lop3_mux<<<(N+BS-1)/BS, BS>>>(d_sel, d_a, d_b, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N*sizeof(unsigned int), cudaMemcpyDeviceToHost);
    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb"); fwrite(h_out, sizeof(unsigned int), N, f); fclose(f);
    }
    bool ok = true;
    for (int i = 0; i < N; i++) {
        unsigned int exp = (h_sel[i]&h_a[i])|(~h_sel[i]&h_b[i]);
        if (h_out[i] != exp) { printf("FAIL at %d\n",i); ok=false; break; }
    }
    printf(ok ? "TEST PASSED\n" : "FAIL\n");
    cudaFree(d_sel); cudaFree(d_a); cudaFree(d_b); cudaFree(d_out);
    return ok ? 0 : 1;
}
