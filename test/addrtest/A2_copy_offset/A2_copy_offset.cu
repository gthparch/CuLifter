extern "C" __global__ void A2_copy_offset(const float* in, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N - 4)
        out[i] = in[i + 4];
}
