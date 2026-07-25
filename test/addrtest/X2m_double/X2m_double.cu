extern "C" __global__ void X2m_double(const double* const* ptrs, double* out,
                                   int N, int which) {
    int i = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (i >= 1 && i < (int)(N - 1)) {
        const double* src = ptrs[which];
        double v0 = src[i + (int)(-1)];
        double v1 = src[i];
        double v2 = src[i + (int)(1)];
        out[i] = v0 + v1 + v2;
    }
}
