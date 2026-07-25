// tma_bounds: Access near array boundaries
// Kernel: out[i] = in[max(0, i-1)] + in[min(N-1, i+1)]
// Tests descriptor addressing with boundary-checked offsets

extern "C" __global__
void tma_bounds(const float* __restrict__ in, float* __restrict__ out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        int left  = (idx > 0)     ? idx - 1 : 0;
        int right = (idx < N - 1) ? idx + 1 : N - 1;
        out[idx] = in[left] + in[right];
    }
}
