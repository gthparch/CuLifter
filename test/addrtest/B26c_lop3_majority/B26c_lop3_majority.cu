extern "C" __global__ void B26c_lop3_majority(const unsigned int* a,
                                   const unsigned int* b,
                                   const unsigned int* c,
                                   unsigned int* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        out[i] = (a[i] & b[i]) | (b[i] & c[i]) | (a[i] & c[i]);
    }
}
