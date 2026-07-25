extern "C" __global__ void H8_signed64_offset(const float* in, float* out,
    long long delta, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        long long idx = (long long)i - delta;
        if (idx >= 0 && idx < (long long)N)
            out[i] = in[idx];
    }
}
