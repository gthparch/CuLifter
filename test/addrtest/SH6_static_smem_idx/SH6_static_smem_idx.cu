extern "C" __global__ void SH6_static_smem_idx(const float* in, float* out, int N) {
    __shared__ float s[64];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        int idx = threadIdx.x % 64;
        s[idx] = in[i];
        out[i] = s[idx];
    }
}
