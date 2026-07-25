extern "C" __global__ void B24d_dual_pred_ternary(const float* a, const float* b,
                                   const float* c, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float va = a[i], vb = b[i], vc = c[i];
        out[i] = (va > vb) ? ((va > vc) ? va : vc) : ((vb > vc) ? vb : vc);
    }
}
