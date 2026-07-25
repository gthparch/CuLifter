
extern "C" __global__ void MU_exp2_raw(const float* in, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) { out[i] = exp2f(in[i]); }
}
