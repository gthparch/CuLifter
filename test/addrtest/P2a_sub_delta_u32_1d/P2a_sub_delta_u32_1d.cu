extern "C" __global__ void P2a_sub_delta_u32_1d(const unsigned int* in, unsigned int* out, int N, int delta) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= delta && i < N)
        out[i] = in[i - delta];
}
