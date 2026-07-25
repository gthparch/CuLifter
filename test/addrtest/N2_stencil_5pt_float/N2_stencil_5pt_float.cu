extern "C" __global__ void addr_N2_stencil_5pt_float(const float* in, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 2 && i < N - 2)
        out[i] = in[i - 2] + in[i - 1] + in[i] + in[i + 1] + in[i + 2];
}
