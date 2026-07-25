extern "C" __global__ void B26a_lop3_and_or(const unsigned int* a,
                                   const unsigned int* b,
                                   const unsigned int* c,
                                   unsigned int* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        out[i] = (a[i] & b[i]) | c[i];
    }
}
