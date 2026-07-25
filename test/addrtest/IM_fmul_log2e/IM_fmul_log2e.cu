extern "C" __global__ void IM_fmul_log2e(const float* in, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float x = in[i];
        float c = 1.4426950408889634f;
        out[i] = x * c;
    }
}
