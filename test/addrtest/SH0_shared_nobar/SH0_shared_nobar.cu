extern "C" __global__ void SH0_shared_nobar(const float* in, float* out, int N) {
    __shared__ float smem[256];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        smem[threadIdx.x] = in[i];
        out[i] = smem[threadIdx.x];
    }
}
