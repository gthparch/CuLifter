
extern "C" __global__ void MU_lg2(const float* in, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) { out[i] = log2f(in[i]); }
}
