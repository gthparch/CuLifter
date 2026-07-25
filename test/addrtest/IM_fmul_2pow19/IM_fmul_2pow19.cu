extern "C" __global__ void IM_fmul_2pow19(const float* in, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float x = in[i];
        float c = 524288.0f;
        out[i] = x * c;
    }
}
