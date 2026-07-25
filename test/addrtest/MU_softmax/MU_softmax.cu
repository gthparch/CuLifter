extern "C" __global__ void MU_softmax(const float* in, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        // Simple per-element softmax over a fixed window of 8
        int base = (i / 8) * 8;
        int end = base + 8; if (end > N) end = N;
        float maxv = in[base];
        for (int j = base+1; j < end; j++) maxv = fmaxf(maxv, in[j]);
        float sum = 0.0f;
        for (int j = base; j < end; j++) sum += expf(in[j] - maxv);
        out[i] = expf(in[i] - maxv) / sum;
    }
}
