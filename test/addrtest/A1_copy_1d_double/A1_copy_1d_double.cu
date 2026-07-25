extern "C" __global__ void addr_A1_copy_1d_double(const double* in, double* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        out[i] = in[i];
}
