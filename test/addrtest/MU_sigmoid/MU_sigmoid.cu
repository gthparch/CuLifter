
extern "C" __global__ void MU_sigmoid(const float* in, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) { out[i] = 1.0f / (1.0f + expf(-in[i])); }
}
