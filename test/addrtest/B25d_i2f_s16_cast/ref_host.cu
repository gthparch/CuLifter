#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

extern "C" __global__ void B25d_i2f_s16_cast(const unsigned int* in, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        unsigned int x = in[i];
        short s = (short)(x & 0xFFFFu);
        out[i] = (float)s;
    }
}

int main(int argc, char** argv) {
    constexpr int N = 1024, BS = 256;
    unsigned int h_in[N]; float h_out[N];
    for (int i = 0; i < N; i++) h_in[i] = (unsigned int)((i*97+13) & 0xFFFFu);
    memset(h_out, 0, sizeof(h_out));
    unsigned int *d_in; float *d_out;
    cudaMalloc(&d_in, N*sizeof(unsigned int)); cudaMalloc(&d_out, N*sizeof(float));
    cudaMemcpy(d_in, h_in, N*sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, N*sizeof(float));
    B25d_i2f_s16_cast<<<(N+BS-1)/BS, BS>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N*sizeof(float), cudaMemcpyDeviceToHost);
    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb"); fwrite(h_out, sizeof(float), N, f); fclose(f);
    }
    bool ok = true;
    for (int i = 0; i < N; i++) {
        float exp = (float)(short)(h_in[i] & 0xFFFFu);
        if (fabsf(h_out[i]-exp) > 1e-5f) { printf("FAIL at %d\n",i); ok=false; break; }
    }
    printf(ok ? "TEST PASSED\n" : "FAIL\n");
    cudaFree(d_in); cudaFree(d_out);
    return ok ? 0 : 1;
}
