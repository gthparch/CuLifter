extern "C" __global__ void IM_fadd_half(const float* in, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float x = in[i];
        float c = 0.5f;
        out[i] = x + c;
    }
}
