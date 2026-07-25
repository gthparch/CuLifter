// tma_desc_scalar: Descriptor + scalar combination
// Kernel: out[i] = in[i] * alpha + beta
// Tests descriptor-based load combined with constant bank parameter access

extern "C" __global__
void tma_desc_scalar(const float* __restrict__ in, float* __restrict__ out,
                     int N, float alpha, float beta) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = in[idx] * alpha + beta;
    }
}
