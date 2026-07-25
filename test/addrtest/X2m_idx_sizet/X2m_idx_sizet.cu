extern "C" __global__ void X2m_idx_sizet(const float* const* ptrs, float* out,
                                   int N, int which) {
    size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (i >= 1 && i < (size_t)(N - 1)) {
        const float* src = ptrs[which];
        float v0 = src[i + (size_t)(-1)];
        float v1 = src[i];
        float v2 = src[i + (size_t)(1)];
        out[i] = v0 + v1 + v2;
    }
}
