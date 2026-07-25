extern "C" __global__ void S_sten3_u8_1d(const unsigned char* in, unsigned char* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 1 && i < N - 1)
        out[i] = (unsigned char)(in[i - 1] + in[i] + in[i + 1]);
}
