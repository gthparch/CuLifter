extern "C" __global__ void S_sten3_u32_1d(const unsigned int* in, unsigned int* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 1 && i < N - 1)
        out[i] = (unsigned int)(in[i - 1] + in[i] + in[i + 1]);
}
