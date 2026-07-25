extern "C" __global__ void M2m_double(const double* const* ptrs, double* out,
                                   int N, int which) {
    int i = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (i < (int)N) {
        const double* src = ptrs[which];
        out[i] = src[i];
    }
}
