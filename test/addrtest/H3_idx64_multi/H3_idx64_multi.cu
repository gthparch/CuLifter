extern "C" __global__ void H3_idx64_multi(const float* in, float* out,
    size_t a, size_t b, size_t c, int N) {
    size_t i = (size_t)threadIdx.x;
    size_t idx = a * i + b * (size_t)blockIdx.x + c;
    int j = (int)(idx % (size_t)N);
    if (j < N) out[j] = in[j];
}
