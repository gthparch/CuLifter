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
