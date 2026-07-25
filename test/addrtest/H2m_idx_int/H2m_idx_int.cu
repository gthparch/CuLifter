extern "C" __global__ void H2m_idx_int(const float* in, float* out, int N) {
    int idx = (int)blockIdx.x * 1048576ULL + (int)threadIdx.x;
    int i = (int)(idx % (int)N);
    if (i < N)
        out[i] = in[i];
}
