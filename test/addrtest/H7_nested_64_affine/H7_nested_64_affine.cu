extern "C" __global__ void H7_nested_64_affine(const float* in, float* out,
    unsigned long long S0, unsigned long long S1, int N) {
    unsigned long long b0 = blockIdx.x;
    unsigned long long b1 = threadIdx.x;
    unsigned long long idx = ((b0 * S0 + b1) * S1 + b1);
    int i = (int)(idx % (unsigned long long)N);
    if (i < N)
        out[i] = in[i];
}
