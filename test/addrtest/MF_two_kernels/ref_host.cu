#include <cstdio>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>
extern "C" __global__ void MF_kernel_simple(const float* in, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = in[i] * 2.0f;
}

extern "C" __global__ void MF_kernel_complex(const float4* in, float4* out, float alpha, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float4 v = in[i];
        v.x = v.x * alpha + 1.0f;
        v.y = v.y * alpha + 2.0f;
        v.z = v.z * alpha + 3.0f;
        v.w = v.w * alpha + 4.0f;
        out[i] = v;
    }
}

int main(int argc, char** argv) {
    constexpr int N = 256;
    float h_in[N], h_out[N];
    for (int i = 0; i < N; i++) h_in[i] = (float)(i+1);
    memset(h_out, 0, sizeof(h_out));
    float *d_in, *d_out;
    cudaMalloc(&d_in, N*4); cudaMalloc(&d_out, N*4);
    cudaMemcpy(d_in, h_in, N*4, cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, N*4);
    MF_kernel_simple<<<(N+255)/256, 256>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N*4, cudaMemcpyDeviceToHost);
    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb"); fwrite(h_out, 4, N, f); fclose(f);
    }
    bool ok = true;
    for (int i = 0; i < N; i++)
        if (fabsf(h_out[i]-h_in[i]*2.0f)>1e-4f){ printf("FAIL %d\n",i);ok=false;break; }
    printf(ok ? "TEST PASSED\n" : "FAIL\n");
    cudaFree(d_in); cudaFree(d_out); return ok ? 0 : 1;
}
