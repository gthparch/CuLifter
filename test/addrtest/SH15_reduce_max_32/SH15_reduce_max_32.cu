extern "C" __global__ void SH15_reduce_max_32(const float* in, float* out, int N) {
    __shared__ float s[32];
    int i = blockIdx.x * 32 + threadIdx.x;
    s[threadIdx.x] = (i < N) ? in[i] : -1e30f;
    __syncthreads();
    for (int stride = 32/2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride && s[threadIdx.x + stride] > s[threadIdx.x])
            s[threadIdx.x] = s[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[blockIdx.x] = s[0];
}
