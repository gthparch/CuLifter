extern "C" __global__ void S_copy_u16_1d(const unsigned short* in, unsigned short* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        out[i] = in[i];
}
