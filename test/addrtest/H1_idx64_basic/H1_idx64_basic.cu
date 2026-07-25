extern "C" __global__ void H1_idx64_basic(const float* in, float* out, int N) {
    size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (idx < (size_t)N)
        out[idx] = in[idx];
}
