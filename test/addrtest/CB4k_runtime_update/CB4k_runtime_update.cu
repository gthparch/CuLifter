__device__ float d_multiplier;

extern "C" __global__ void CB4k_runtime_update(const float* in, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = in[i] * d_multiplier;
}
