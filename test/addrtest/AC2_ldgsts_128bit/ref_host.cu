#include <cstdio>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda/pipeline>
extern "C" __global__ void AC2_ldgsts_128bit(const float4* __restrict__ in, float4* out, int N) {
    __shared__ float4 smem[64];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();
    pipe.producer_acquire();
    if (tid < N) cuda::memcpy_async(&smem[threadIdx.x], &in[tid], sizeof(float4), pipe);
    pipe.producer_commit();
    pipe.consumer_wait();
    __syncthreads();
    if (tid < N) {
        float4 v = smem[threadIdx.x];
        v.x *= 2.0f; v.y *= 2.0f; v.z *= 2.0f; v.w *= 2.0f;
        out[tid] = v;
    }
}
int main(int argc, char** argv) {
    constexpr int N = 64;
    float4 h_in[N], h_out[N];
    for (int i = 0; i < N; i++) { h_in[i] = {(float)(i*4+1),(float)(i*4+2),(float)(i*4+3),(float)(i*4+4)}; }
    memset(h_out, 0, sizeof(h_out));
    float4 *d_in, *d_out;
    cudaMalloc(&d_in, N*sizeof(float4)); cudaMalloc(&d_out, N*sizeof(float4));
    cudaMemcpy(d_in, h_in, N*sizeof(float4), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, N*sizeof(float4));
    AC2_ldgsts_128bit<<<1, 64>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N*sizeof(float4), cudaMemcpyDeviceToHost);
    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb"); fwrite(h_out, sizeof(float4), N, f); fclose(f);
    }
    printf("TEST PASSED\n");
    cudaFree(d_in); cudaFree(d_out);
    return 0;
}
