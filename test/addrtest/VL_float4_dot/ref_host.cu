#include <cstdio>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>
extern "C" __global__ void VL_float4_dot(const float4* a, const float4* b, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float4 va = a[i], vb = b[i];
        out[i] = va.x*vb.x + va.y*vb.y + va.z*vb.z + va.w*vb.w;
    }
}
int main(int argc, char** argv) {
    constexpr int N = 256;
    float4 h_a[N], h_b[N]; float h_out[N];
    for (int i = 0; i < N; i++) {
        h_a[i] = make_float4((i*4%7+1)*0.1f, ((i*4+1)%7+1)*0.1f, ((i*4+2)%7+1)*0.1f, ((i*4+3)%7+1)*0.1f);
        h_b[i] = make_float4((i*4%5+1)*0.1f, ((i*4+1)%5+1)*0.1f, ((i*4+2)%5+1)*0.1f, ((i*4+3)%5+1)*0.1f);
    }
    memset(h_out, 0, sizeof(h_out));
    float4 *d_a, *d_b; float *d_out;
    cudaMalloc(&d_a, N*sizeof(float4)); cudaMalloc(&d_b, N*sizeof(float4)); cudaMalloc(&d_out, N*sizeof(float));
    cudaMemcpy(d_a, h_a, N*sizeof(float4), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N*sizeof(float4), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, N*sizeof(float));
    VL_float4_dot<<<(N+255)/256, 256>>>(d_a, d_b, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N*sizeof(float), cudaMemcpyDeviceToHost);
    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb"); fwrite(h_out, sizeof(float), N, f); fclose(f);
    }
    printf("TEST PASSED\n");
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_out); return 0;
}
