extern "C" __global__ void S_sten5_f32_1d(const float* in, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 2 && i < N - 2)
        out[i] = (float)(in[i - 2] + in[i - 1] + in[i] + in[i + 1] + in[i + 2]);
}
