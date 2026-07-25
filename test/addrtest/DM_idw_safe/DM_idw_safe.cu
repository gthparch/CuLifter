extern "C" __global__ void DM_idw_safe(const float* x, const float* y,
    const float* vals, float* out, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    float sum_w = 0.0f, sum_wv = 0.0f;
    for (int i = 0; i < N; i++) {
        if (i == tid) continue;
        float dx = x[tid] - x[i], dy = y[tid] - y[i];
        float d = sqrtf(dx*dx + dy*dy + 1e-8f);
        float w = 1.0f / d;
        sum_w += w; sum_wv += w * vals[i];
    }
    out[tid] = sum_wv / (sum_w + 1e-8f);
}
