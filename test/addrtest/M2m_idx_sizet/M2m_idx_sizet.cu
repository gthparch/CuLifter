extern "C" __global__ void M2m_idx_sizet(const float* const* ptrs, float* out,
                                   int N, int which) {
    size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (i < (size_t)N) {
        const float* src = ptrs[which];
        out[i] = src[i];
    }
}
