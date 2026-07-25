extern "C" __global__ void B25e_prmt_shuffle(const unsigned int* a,
                                   const unsigned int* b,
                                   unsigned int* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        out[i] = __byte_perm(a[i], b[i], 0x05040100u);
    }
}
