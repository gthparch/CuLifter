
extern "C" __global__ void MU_exp_edge(const float* in, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) { out[i] = expf(in[i]); }
}
