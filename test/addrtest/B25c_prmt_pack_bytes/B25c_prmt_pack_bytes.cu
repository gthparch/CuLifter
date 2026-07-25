extern "C" __global__ void B25c_prmt_pack_bytes(const unsigned char* in,
                                   unsigned int* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        int base = i * 4;
        unsigned int b0 = (base < N*4)   ? in[base]   : 0u;
        unsigned int b1 = (base+1 < N*4) ? in[base+1] : 0u;
        unsigned int b2 = (base+2 < N*4) ? in[base+2] : 0u;
        unsigned int b3 = (base+3 < N*4) ? in[base+3] : 0u;
        out[i] = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
    }
}
