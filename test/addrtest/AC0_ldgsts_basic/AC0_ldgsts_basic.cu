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
