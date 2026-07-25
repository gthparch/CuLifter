extern "C" __global__ void V_lop3_nand(const unsigned* a, const unsigned* b,
    const unsigned* c, unsigned* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = (unsigned)(~(a[i] & b[i]) & c[i]);
}
