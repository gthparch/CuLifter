extern "C" __global__ void S_copy_u8_1d(const unsigned char* in, unsigned char* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        out[i] = in[i];
}
