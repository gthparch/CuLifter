// tma_two_desc: Two separate array accesses (testing multi-descriptor)
// Kernel: out[i] = a[i] + b[i]

extern "C" __global__
void tma_two_desc(const float* __restrict__ a, const float* __restrict__ b,
                  float* __restrict__ out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = a[idx] + b[idx];
    }
}
