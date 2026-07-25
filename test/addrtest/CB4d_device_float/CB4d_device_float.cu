__device__ float d_scale = 2.5f;

extern "C" __global__ void CB4d_device_float(const float* in, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = in[i] * d_scale;
}
