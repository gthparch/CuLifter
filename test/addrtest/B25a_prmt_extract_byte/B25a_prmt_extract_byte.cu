extern "C" __global__ void B25a_prmt_extract_byte(const unsigned int* in,
                                   unsigned int* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        unsigned int x = in[i];
        unsigned int b0 = x & 0xFFu;
        unsigned int b1 = (x >> 8) & 0xFFu;
        unsigned int b2 = (x >> 16) & 0xFFu;
        unsigned int b3 = (x >> 24) & 0xFFu;
        out[i] = b0 + b1 + b2 + b3;
    }
}
