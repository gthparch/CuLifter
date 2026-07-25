extern "C" __global__ void B28d_int_divmod_combo(const unsigned int* in,
                                   unsigned int* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        unsigned int x = in[i];
        unsigned int q = x / 7u;
        unsigned int r = x % 7u;
        out[i] = q * 7u + r;
    }
}
