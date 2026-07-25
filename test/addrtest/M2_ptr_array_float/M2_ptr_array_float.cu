extern "C" __global__ void addr_M2_ptr_array_float(const float* const* ptrs, float* out,
                                    int N, int which) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        const float* src = ptrs[which];
        out[i] = src[i];
    }
}
