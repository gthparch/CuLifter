extern "C" __global__ void addr_X2_ptr_stencil_float(const float* const* ptrs, float* out,
                                    int N, int which) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 1 && i < N - 1) {
        const float* src = ptrs[which];
        out[i] = src[i - 1] + src[i] + src[i + 1];
    }
}
