extern "C" __global__ void B24b_dual_pred_nested(const int* flags, const float* in, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        int flag = flags[i];
        if (flag) {
            if (i < N) out[i] = in[i] * 2.0f;
        } else {
            out[i] = in[i];
        }
    }
}
