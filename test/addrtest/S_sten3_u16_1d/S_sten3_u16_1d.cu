extern "C" __global__ void S_sten3_u16_1d(const unsigned short* in, unsigned short* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 1 && i < N - 1)
        out[i] = (unsigned short)(in[i - 1] + in[i] + in[i + 1]);
}
