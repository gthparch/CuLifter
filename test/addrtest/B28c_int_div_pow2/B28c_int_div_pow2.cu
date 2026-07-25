extern "C" __global__ void B28c_int_div_pow2(const unsigned int* in,
                                   unsigned int* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        unsigned int div8 = in[i] / 8u;
        unsigned int shr3 = in[i] >> 3;
        out[i] = div8 + shr3;
    }
}
