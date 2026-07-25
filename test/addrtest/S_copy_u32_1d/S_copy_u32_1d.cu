extern "C" __global__ void S_copy_u32_1d(const unsigned int* in, unsigned int* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        out[i] = in[i];
}
