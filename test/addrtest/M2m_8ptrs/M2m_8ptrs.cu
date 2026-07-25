extern "C" __global__ void M2m_8ptrs(const float* const* ptrs, float* out,
                                   int N, int which) {
    int i = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (i < (int)N) {
        const float* src = ptrs[which];
        out[i] = src[i];
    }
}
