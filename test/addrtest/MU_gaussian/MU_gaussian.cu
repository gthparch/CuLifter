
extern "C" __global__ void MU_gaussian(const float* in, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) { float sigma = 2.0f; out[i] = expf(-(in[i]*in[i]) / (2.0f*sigma*sigma)); }
}
