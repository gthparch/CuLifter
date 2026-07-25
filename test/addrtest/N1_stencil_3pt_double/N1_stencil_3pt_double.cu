extern "C" __global__ void addr_N1_stencil_3pt_double(const double* in, double* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 1 && i < N - 1)
        out[i] = in[i - 1] + in[i] + in[i + 1];
}
