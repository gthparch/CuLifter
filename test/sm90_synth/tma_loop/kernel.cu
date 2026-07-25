// tma_loop: Descriptor access in a reduction loop
// Kernel: out[i] = sum of in[i*K .. i*K+K-1]
// Tests descriptor addressing with loop-varying offsets

extern "C" __global__
void tma_loop(const float* __restrict__ in, float* __restrict__ out,
              int N, int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float sum = 0.0f;
        int base = idx * K;
        for (int j = 0; j < K; j++) {
            sum += in[base + j];
        }
        out[idx] = sum;
    }
}
