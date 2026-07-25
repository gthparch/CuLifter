__device__ float d_bias = 100.0f;

extern "C" __global__ void CB4f_mixed_c0_c4(const float* in, float* out, int N, float scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = in[i] * scale + d_bias;
}
