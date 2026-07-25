extern "C" __global__ void P2_ptr_metadata(const float* in, float* out,
    int N, int stride, int offset, int scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        int src_idx = i * stride + offset;
        if (src_idx >= 0 && src_idx < N)
            out[i] = in[src_idx] * (float)scale;
    }
}
