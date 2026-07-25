extern "C" __global__ void S_sten3_f32_1d(const float* in, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 1 && i < N - 1)
        out[i] = (float)(in[i - 1] + in[i] + in[i + 1]);
}
