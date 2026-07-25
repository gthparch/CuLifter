extern "C" __global__ void X2m_radius4(const float* const* ptrs, float* out,
                                   int N, int which) {
    int i = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (i >= 4 && i < (int)(N - 4)) {
        const float* src = ptrs[which];
        float v0 = src[i + (int)(-4)];
        float v1 = src[i + (int)(-3)];
        float v2 = src[i + (int)(-2)];
        float v3 = src[i + (int)(-1)];
        float v4 = src[i];
        float v5 = src[i + (int)(1)];
        float v6 = src[i + (int)(2)];
        float v7 = src[i + (int)(3)];
        float v8 = src[i + (int)(4)];
        out[i] = v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7 + v8;
    }
}
