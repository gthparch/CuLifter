__device__ float d_lut[1024];

extern "C" __global__ void CB4i_large_array(float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = d_lut[i] + 1.0f;
}
