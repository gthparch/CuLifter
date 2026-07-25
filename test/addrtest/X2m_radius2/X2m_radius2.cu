extern "C" __global__ void X2m_radius2(const float* const* ptrs, float* out,
                                   int N, int which) {
    int i = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (i >= 2 && i < (int)(N - 2)) {
        const float* src = ptrs[which];
        float v0 = src[i + (int)(-2)];
        float v1 = src[i + (int)(-1)];
        float v2 = src[i];
        float v3 = src[i + (int)(1)];
        float v4 = src[i + (int)(2)];
        out[i] = v0 + v1 + v2 + v3 + v4;
    }
}
