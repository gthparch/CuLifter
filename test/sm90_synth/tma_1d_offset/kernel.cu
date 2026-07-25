// tma_1d_offset: Simple 1D array access with an offset
// Kernel: out[i] = in[i + offset]
// Tests descriptor-based LDG with non-zero offset

extern "C" __global__
void tma_1d_offset(const float* __restrict__ in, float* __restrict__ out,
                   int N, int offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        int src = idx + offset;
        if (src >= 0 && src < N)
            out[idx] = in[src];
        else
            out[idx] = 0.0f;
    }
}
