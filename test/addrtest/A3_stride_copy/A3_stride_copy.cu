extern "C" __global__ void A3_stride_copy(const float* in, float* out, int N, int stride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N && i * stride < N)
        out[i] = in[i * stride];
}
