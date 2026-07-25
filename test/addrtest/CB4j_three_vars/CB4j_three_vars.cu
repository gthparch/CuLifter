__device__ int d_count = 100;
__device__ float d_scale = 0.5f;
__device__ double d_bias = 3.14159;

extern "C" __global__ void CB4j_three_vars(const float* in, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N && i < d_count) out[i] = in[i] * d_scale + (float)d_bias;
}
