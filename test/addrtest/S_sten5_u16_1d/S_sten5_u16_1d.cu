extern "C" __global__ void S_sten5_u16_1d(const unsigned short* in, unsigned short* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 2 && i < N - 2)
        out[i] = (unsigned short)(in[i - 2] + in[i - 1] + in[i] + in[i + 1] + in[i + 2]);
}
