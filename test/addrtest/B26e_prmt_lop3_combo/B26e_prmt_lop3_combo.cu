extern "C" __global__ void B26e_prmt_lop3_combo(const unsigned int* in,
                                   unsigned int* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        unsigned int x = in[i];
        unsigned int b0 = x & 0xFFu;
        unsigned int b1 = (x >> 8) & 0xFFu;
        unsigned int b2 = (x >> 16) & 0xFFu;
        unsigned int logic = (b0 & b1) | (b1 & b2) | (b0 & b2);
        out[i] = logic;
    }
}
