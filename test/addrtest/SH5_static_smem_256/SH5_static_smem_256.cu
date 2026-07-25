extern "C" __global__ void SH5_static_smem_256(const float* in, float* out, int N) {
    __shared__ float s[256];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        s[threadIdx.x] = in[i];
        out[i] = s[threadIdx.x];
    }
}
