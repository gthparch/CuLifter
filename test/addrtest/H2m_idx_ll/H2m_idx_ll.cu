extern "C" __global__ void H2m_idx_ll(const float* in, float* out, int N) {
    long long idx = (long long)blockIdx.x * 1048576LL + (long long)threadIdx.x;
    int i = (int)(idx % (long long)N);
    if (i < N)
        out[i] = in[i];
}
