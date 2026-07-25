extern "C" __global__ void B24c_dual_pred_or(const float* in, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int x = i - 128;
    if (x < 0 || x >= N) return;
    out[x] = in[x] + 1.0f;
}
