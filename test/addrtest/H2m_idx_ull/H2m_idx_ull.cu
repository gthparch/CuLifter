extern "C" __global__ void H2m_idx_ull(const float* in, float* out, int N) {
    unsigned long long idx = (unsigned long long)blockIdx.x * 1048576ULL + (unsigned long long)threadIdx.x;
    int i = (int)(idx % (unsigned long long)N);
    if (i < N)
        out[i] = in[i];
}
