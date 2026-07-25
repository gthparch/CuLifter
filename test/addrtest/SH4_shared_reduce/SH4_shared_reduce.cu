extern "C" __global__ void SH4_shared_reduce(const float* in, float* out, int N) {
    __shared__ float smem[256];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    smem[threadIdx.x] = (i < N) ? in[i] : 0.0f;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        out[blockIdx.x] = smem[0];
}
