extern "C" __global__ void addr_H2_big_mul_sizet(const float* in, float* out, int N) {
    size_t idx = (size_t)blockIdx.x * 1048576ULL + (size_t)threadIdx.x;
    // Map back to a small range for actual access
    int i = (int)(idx % (size_t)N);
    if (i < N)
        out[i] = in[i];
}
