extern "C" __global__ void B25b_prmt_extract_half(const unsigned int* in, int* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        unsigned int x = in[i];
        short lo = (short)(x & 0xFFFFu);
        short hi = (short)(x >> 16);
        out[i] = (int)lo + (int)hi;
    }
}
