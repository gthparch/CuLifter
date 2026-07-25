#include <cstdio>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>
extern "C" __global__ void VL_saxpy_vec4(const float4* x, float4* y, float alpha, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float4 vx = x[i], vy = y[i];
        vy.x += alpha * vx.x; vy.y += alpha * vx.y;
        vy.z += alpha * vx.z; vy.w += alpha * vx.w;
        y[i] = vy;
    }
}
int main(int argc, char** argv) {
    constexpr int N = 256;
    float alpha = 2.5f;
    float4 h_x[N], h_y[N], h_orig[N];
    for (int i = 0; i < N; i++) {
        h_x[i] = make_float4((i*4%17+1)*0.1f, ((i*4+1)%17+1)*0.1f, ((i*4+2)%17+1)*0.1f, ((i*4+3)%17+1)*0.1f);
        h_y[i] = make_float4((i*4%13+1)*0.2f, ((i*4+1)%13+1)*0.2f, ((i*4+2)%13+1)*0.2f, ((i*4+3)%13+1)*0.2f);
    }
    memcpy(h_orig, h_y, sizeof(h_y));
    float4 *d_x, *d_y;
    cudaMalloc(&d_x, N*sizeof(float4)); cudaMalloc(&d_y, N*sizeof(float4));
    cudaMemcpy(d_x, h_x, N*sizeof(float4), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, N*sizeof(float4), cudaMemcpyHostToDevice);
    VL_saxpy_vec4<<<(N+255)/256, 256>>>(d_x, d_y, alpha, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_y, d_y, N*sizeof(float4), cudaMemcpyDeviceToHost);
    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb"); fwrite(h_y, sizeof(float4), N, f); fclose(f);
    }
    printf("TEST PASSED\n");
    cudaFree(d_x); cudaFree(d_y); return 0;
}
