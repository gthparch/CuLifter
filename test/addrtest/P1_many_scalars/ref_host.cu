#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

extern "C" __global__ void P1_many_scalars(const float* in, float* out, int N,
    float a0, float a1, float a2, float a3, float a4, float a5, float a6, float a7) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        out[i] = in[i] * a0 + a1 + a2 - a3 + a4 * a5 + a6 - a7;
}

int main(int argc, char** argv) {
    constexpr int N = 1024;
    float h_in[N], h_out[N];
    memset(h_out, 0, sizeof(h_out));
    for (int i = 0; i < N; i++) h_in[i] = (float)(i%7+1)*0.1f;
    float a0=1.0f,a1=0.1f,a2=0.2f,a3=0.3f,a4=0.4f,a5=0.5f,a6=0.6f,a7=0.7f;
    float *d_in, *d_out;
    cudaMalloc(&d_in, N*sizeof(float)); cudaMalloc(&d_out, N*sizeof(float));
    cudaMemcpy(d_in, h_in, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, N*sizeof(float));
    P1_many_scalars<<<(N+255)/256, 256>>>(d_in, d_out, N, a0,a1,a2,a3,a4,a5,a6,a7);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N*sizeof(float), cudaMemcpyDeviceToHost);
    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb");
        fwrite(h_out, sizeof(h_out[0]), sizeof(h_out)/sizeof(h_out[0]), f);
        fclose(f);
    }
    bool ok = true;
    for (int i = 0; i < N; i++) {
        float e = h_in[i]*a0+a1+a2-a3+a4*a5+a6-a7;
        if (fabsf(h_out[i]-e) > 1e-4f) { printf("FAIL at %d\n",i); ok=false; break; }
    }
    printf(ok ? "TEST PASSED\n" : "FAIL\n");
    cudaFree(d_in); cudaFree(d_out);
    return ok ? 0 : 1;
}
