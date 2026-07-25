extern "C" __global__ void H2m_double(const double* in, double* out, int N) {
    size_t idx = (size_t)blockIdx.x * 1048576ULL + (size_t)threadIdx.x;
    int i = (int)(idx % (size_t)N);
    if (i < N)
        out[i] = in[i];
}
