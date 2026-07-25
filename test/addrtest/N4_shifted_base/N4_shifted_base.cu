extern "C" __global__ void N4_shifted_base(const float* in, float* out, int N, int shift) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    const float* p = in + shift;
    if (i >= 1 && i < N - shift - 1)
        out[i] = p[i - 1] + p[i] + p[i + 1];
}
