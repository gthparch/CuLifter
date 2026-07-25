extern "C" __global__ void IM_fma_negpi2(const float* in, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float x = in[i];
        float c = -1.570796251296997f;
        out[i] = x * c + 1.0f;
    }
}
