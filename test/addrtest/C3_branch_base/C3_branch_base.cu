extern "C" __global__ void C3_branch_base(const float* in1, const float* in2,
    float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        const float* src;
        if (i % 2 == 0) src = in1; else src = in2;
        out[i] = src[i];
    }
}
