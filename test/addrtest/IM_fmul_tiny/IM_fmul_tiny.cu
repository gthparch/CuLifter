extern "C" __global__ void IM_fmul_tiny(const float* in, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float x = in[i];
        float c = 1.925963e-08f;
        out[i] = x * c + 1.0f;
    }
}
