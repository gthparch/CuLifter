extern "C" __global__ void MU_exp_chain(const float* a, const float* b, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = expf(a[i]) * expf(b[i]);
}
