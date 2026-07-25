extern "C" __global__ void H5_u64_compose(const float* in, float* out,
    unsigned int hi, unsigned int lo, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long off = ((unsigned long long)hi << 32) | (unsigned long long)lo;
    if (i < N) {
        // off is 0 in practice but forces hi/lo composition
        out[i] = in[i + (int)(off % (unsigned long long)N)];
    }
}
