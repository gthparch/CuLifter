extern "C" __global__ void IM_fadd_magic(const float* in, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float x = in[i];
        float c = 12582912.0f;
        out[i] = x + c;
    }
}
