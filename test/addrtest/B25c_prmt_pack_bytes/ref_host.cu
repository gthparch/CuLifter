#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

extern "C" __global__ void B25c_prmt_pack_bytes(const unsigned char* in,
                                   unsigned int* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        int base = i * 4;
        unsigned int b0 = (base < N*4)   ? in[base]   : 0u;
        unsigned int b1 = (base+1 < N*4) ? in[base+1] : 0u;
        unsigned int b2 = (base+2 < N*4) ? in[base+2] : 0u;
        unsigned int b3 = (base+3 < N*4) ? in[base+3] : 0u;
        out[i] = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
    }
}

int main(int argc, char** argv) {
    constexpr int N = 1024, BS = 256;
    unsigned char h_in[N*4]; unsigned int h_out[N];
    for (int i = 0; i < N*4; i++) h_in[i] = (unsigned char)((i*7+3)%256);
    memset(h_out, 0, sizeof(h_out));
    unsigned char *d_in; unsigned int *d_out;
    cudaMalloc(&d_in, N*4*sizeof(unsigned char)); cudaMalloc(&d_out, N*sizeof(unsigned int));
    cudaMemcpy(d_in, h_in, N*4*sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, N*sizeof(unsigned int));
    B25c_prmt_pack_bytes<<<(N+BS-1)/BS, BS>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N*sizeof(unsigned int), cudaMemcpyDeviceToHost);
    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb"); fwrite(h_out, sizeof(unsigned int), N, f); fclose(f);
    }
    bool ok = true;
    for (int i = 0; i < N; i++) {
        int base = i*4;
        unsigned int exp = (unsigned int)h_in[base] | ((unsigned int)h_in[base+1]<<8)
                         | ((unsigned int)h_in[base+2]<<16) | ((unsigned int)h_in[base+3]<<24);
        if (h_out[i] != exp) { printf("FAIL at %d\n",i); ok=false; break; }
    }
    printf(ok ? "TEST PASSED\n" : "FAIL\n");
    cudaFree(d_in); cudaFree(d_out);
    return ok ? 0 : 1;
}
