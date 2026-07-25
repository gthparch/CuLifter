extern "C" __global__ void B25d_i2f_s16_cast(const unsigned int* in, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        unsigned int x = in[i];
        short s = (short)(x & 0xFFFFu);
        out[i] = (float)s;
    }
}
