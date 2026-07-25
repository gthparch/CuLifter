__device__ int d_offset = 10;
__device__ float d_factor = 3.0f;

extern "C" __global__ void CB4e_two_device_vars(const float* in, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        int idx = i + d_offset;
        if (idx < N) out[i] = in[idx] * d_factor;
        else out[i] = 0.0f;
    }
}
