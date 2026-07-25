#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

extern "C" __global__ void F3s_i2f_u8(const unsigned char* in, half* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        out[i] = __ushort2half_rn((unsigned short)in[i]);
}

int main(int argc, char** argv) {
    constexpr int N = 1024;
    unsigned char h_in[N]; half h_out[N];
    for (int i = 0; i < N; i++) h_in[i] = (unsigned char)(i%256);
    memset(h_out, 0, sizeof(h_out));
    unsigned char *d_in; half *d_out;
    cudaMalloc(&d_in, N*sizeof(unsigned char)); cudaMalloc(&d_out, N*sizeof(half));
    cudaMemcpy(d_in, h_in, N*sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, N*sizeof(half));
    F3s_i2f_u8<<<(N+255)/256, 256>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N*sizeof(half), cudaMemcpyDeviceToHost);
    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb"); fwrite(h_out, sizeof(half), N, f); fclose(f);
    }
    bool ok = true;
    for (int i = 0; i < N; i++) {
        float got = __half2float(h_out[i]);
        if (fabsf(got-(float)h_in[i]) > 0.5f) { printf("FAIL at %d\n",i); ok=false; break; }
    }
    printf(ok ? "TEST PASSED\n" : "FAIL\n");
    cudaFree(d_in); cudaFree(d_out);
    return ok ? 0 : 1;
}
