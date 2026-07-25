#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

extern "C" __global__ void B25e_prmt_shuffle(const unsigned int* a,
                                   const unsigned int* b,
                                   unsigned int* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        out[i] = __byte_perm(a[i], b[i], 0x05040100u);
    }
}

int main(int argc, char** argv) {
    constexpr int N = 1024, BS = 256;
    unsigned int h_a[N], h_b[N], h_out[N];
    for (int i = 0; i < N; i++) { h_a[i] = (unsigned int)(i*0x01030507u+1); h_b[i] = (unsigned int)(i*0x02040608u+3); }
    memset(h_out, 0, sizeof(h_out));
    unsigned int *d_a, *d_b, *d_out;
    cudaMalloc(&d_a, N*sizeof(unsigned int)); cudaMalloc(&d_b, N*sizeof(unsigned int));
    cudaMalloc(&d_out, N*sizeof(unsigned int));
    cudaMemcpy(d_a, h_a, N*sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N*sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, N*sizeof(unsigned int));
    B25e_prmt_shuffle<<<(N+BS-1)/BS, BS>>>(d_a, d_b, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N*sizeof(unsigned int), cudaMemcpyDeviceToHost);
    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb"); fwrite(h_out, sizeof(unsigned int), N, f); fclose(f);
    }
    // CPU byte_perm
    bool ok = true;
    for (int i = 0; i < N; i++) {
        unsigned char src[8];
        unsigned int aa = h_a[i], bb = h_b[i];
        src[0]=(unsigned char)(aa); src[1]=(unsigned char)(aa>>8);
        src[2]=(unsigned char)(aa>>16); src[3]=(unsigned char)(aa>>24);
        src[4]=(unsigned char)(bb); src[5]=(unsigned char)(bb>>8);
        src[6]=(unsigned char)(bb>>16); src[7]=(unsigned char)(bb>>24);
        unsigned int sel = 0x05040100u, r = 0;
        for (int j=0;j<4;j++) { unsigned int idx=(sel>>(j*4))&0x7u; r|=((unsigned int)src[idx])<<(j*8); }
        if (h_out[i] != r) { printf("FAIL at %d\n",i); ok=false; break; }
    }
    printf(ok ? "TEST PASSED\n" : "FAIL\n");
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_out);
    return ok ? 0 : 1;
}
