extern "C" __global__ void P2a_sub_delta_f32_1d(const float* in, float* out, int N, int delta) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= delta && i < N)
        out[i] = in[i - delta];
}
