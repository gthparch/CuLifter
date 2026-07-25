extern "C" __global__ void B28b_int_mod_const(const unsigned int* in,
                                   unsigned int* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        out[i] = in[i] % 7u;
    }
}
