extern "C" __global__ void B26d_lop3_mux(const unsigned int* sel,
                                   const unsigned int* a,
                                   const unsigned int* b,
                                   unsigned int* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        out[i] = (sel[i] & a[i]) | (~sel[i] & b[i]);
    }
}
