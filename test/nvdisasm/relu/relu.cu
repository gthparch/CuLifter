#include <cuda_runtime.h>

__global__ void relu(const float* __restrict__ in,
                     float* __restrict__ out,
                     int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = in[i];
        out[i] = v > 0.0f ? v : 0.0f;
    }
}