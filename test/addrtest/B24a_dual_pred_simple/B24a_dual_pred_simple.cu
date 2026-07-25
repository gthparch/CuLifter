extern "C" __global__ void B24a_dual_pred_simple(const float* in, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float x = (float)i - (float)(N/2);
    if (x > 0.0f && x < (float)N) {
        out[i] = in[i] + x;
    } else {
        if (i < N) out[i] = in[i];
    }
}
