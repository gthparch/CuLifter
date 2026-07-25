extern "C" __global__ void addr_H2_big_mul_int(const float* in, float* out, int N) {
    int idx = (int)blockIdx.x * 1048576ULL + (int)threadIdx.x;
    // Map back to a small range for actual access
    int i = (int)(idx % (int)N);
    if (i < N)
        out[i] = in[i];
}
