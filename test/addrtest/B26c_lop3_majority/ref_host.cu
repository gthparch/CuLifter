#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

extern "C" __global__ void B26c_lop3_majority(const unsigned int* a,
                                   const unsigned int* b,
                                   const unsigned int* c,
                                   unsigned int* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        out[i] = (a[i] & b[i]) | (b[i] & c[i]) | (a[i] & c[i]);
    }
}

int main(int argc, char** argv) {
    constexpr int N = 1024, BS = 256;
    unsigned int h_a[N], h_b[N], h_c[N], h_out[N];
    for (int i = 0; i < N; i++) {
        h_a[i] = (unsigned int)(i*0xAAAAAAAAu+1);
        h_b[i] = (unsigned int)(i*0x55555555u+3);
        h_c[i] = (unsigned int)(i*0x33333333u+5);
    }
    memset(h_out, 0, sizeof(h_out));
    unsigned int *d_a, *d_b, *d_c, *d_out;
    cudaMalloc(&d_a, N*sizeof(unsigned int)); cudaMalloc(&d_b, N*sizeof(unsigned int));
    cudaMalloc(&d_c, N*sizeof(unsigned int)); cudaMalloc(&d_out, N*sizeof(unsigned int));
    cudaMemcpy(d_a, h_a, N*sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N*sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, N*sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, N*sizeof(unsigned int));
    B26c_lop3_majority<<<(N+BS-1)/BS, BS>>>(d_a, d_b, d_c, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N*sizeof(unsigned int), cudaMemcpyDeviceToHost);
    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb"); fwrite(h_out, sizeof(unsigned int), N, f); fclose(f);
    }
    bool ok = true;
    for (int i = 0; i < N; i++) {
        unsigned int exp = (h_a[i]&h_b[i])|(h_b[i]&h_c[i])|(h_a[i]&h_c[i]);
        if (h_out[i] != exp) { printf("FAIL at %d\n",i); ok=false; break; }
    }
    printf(ok ? "TEST PASSED\n" : "FAIL\n");
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); cudaFree(d_out);
    return ok ? 0 : 1;
}
