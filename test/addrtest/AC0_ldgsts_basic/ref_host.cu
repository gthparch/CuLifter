#include <cstdio>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda/pipeline>
extern "C" __global__ void AC0_ldgsts_basic(const float* __restrict__ in, float* out, int N) {
    __shared__ float smem[256];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();
    pipe.producer_acquire();
    if (tid < N) cuda::memcpy_async(&smem[threadIdx.x], &in[tid], sizeof(float), pipe);
    pipe.producer_commit();
    pipe.consumer_wait();
    __syncthreads();
    if (tid < N) out[tid] = smem[threadIdx.x] * 2.0f;
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
    AC0_ldgsts_basic<<<1, 256>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N*4, cudaMemcpyDeviceToHost);
    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb"); fwrite(h_out, 4, N, f); fclose(f);
    }
    bool ok = true;
    for (int i = 0; i < N; i++)
        if (fabsf(h_out[i]-h_in[i]*2.0f) > 1e-5f) { printf("FAIL %d\n", i); ok=false; break; }
    printf(ok ? "TEST PASSED\n" : "FAIL\n");
    cudaFree(d_in); cudaFree(d_out);
    return ok ? 0 : 1;
}
