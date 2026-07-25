extern "C" __global__ void P1_many_scalars(const float* in, float* out, int N,
    float a0, float a1, float a2, float a3, float a4, float a5, float a6, float a7) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        out[i] = in[i] * a0 + a1 + a2 - a3 + a4 * a5 + a6 - a7;
}
