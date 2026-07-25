extern "C" __global__ void M2m_dbl_indirect(const float* const* const* ptr_tables,
                                   float* out, int N, int table, int which) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        const float* const* ptrs = ptr_tables[table];
        const float* src = ptrs[which];
        out[i] = src[i];
    }
}
