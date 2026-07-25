extern "C" __global__ void P2d_load_sub_f32(const float* data, const int* offsets,
    float* out, int N, int K) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        int off = offsets[i];
        int idx = off - K;
        if (idx >= 0 && idx < N)
            out[i] = data[idx];
    }
}
