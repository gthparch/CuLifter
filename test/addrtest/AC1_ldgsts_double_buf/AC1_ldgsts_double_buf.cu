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
