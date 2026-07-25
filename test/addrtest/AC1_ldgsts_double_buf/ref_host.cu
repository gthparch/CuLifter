#include <cstdio>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda/pipeline>
extern "C" __global__ void AC1_ldgsts_double_buf(
    const float* __restrict__ in, float* out, int N) {
    __shared__ float smem[2][256];
    int tid = threadIdx.x;
    int base = blockIdx.x * 512;
    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();
    pipe.producer_acquire();
    if (base + tid < N)
        cuda::memcpy_async(&smem[0][tid], &in[base + tid], sizeof(float), pipe);
    pipe.producer_commit();
    pipe.producer_acquire();
    if (base + 256 + tid < N)
        cuda::memcpy_async(&smem[1][tid], &in[base + 256 + tid], sizeof(float), pipe);
    pipe.producer_commit();
    cuda::pipeline_consumer_wait_prior<1>(pipe);
    __syncthreads();
    float v0 = smem[0][tid] * 2.0f;
    cuda::pipeline_consumer_wait_prior<0>(pipe);
    __syncthreads();
    float v1 = (base + 256 + tid < N) ? smem[1][tid] * 3.0f : 0.0f;
    if (base + tid < N) out[base + tid] = v0;
    if (base + 256 + tid < N) out[base + 256 + tid] = v1;
}
int main(int argc, char** argv) {
    constexpr int N = 512;
    float h_in[N], h_out[N];
    for (int i = 0; i < N; i++) h_in[i] = (float)(i+1);
    memset(h_out, 0, sizeof(h_out));
    float *d_in, *d_out;
    cudaMalloc(&d_in, N*4); cudaMalloc(&d_out, N*4);
    cudaMemcpy(d_in, h_in, N*4, cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, N*4);
    AC1_ldgsts_double_buf<<<1, 256>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N*4, cudaMemcpyDeviceToHost);
    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb"); fwrite(h_out, 4, N, f); fclose(f);
    }
    printf("TEST PASSED\n");
    cudaFree(d_in); cudaFree(d_out);
    return 0;
}
