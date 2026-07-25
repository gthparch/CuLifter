extern "C" __global__ void SH14_reduce_sum_16(const float* in, float* out, int N) {
    __shared__ float s[16];
    int i = blockIdx.x * 16 + threadIdx.x;
    s[threadIdx.x] = (i < N) ? in[i] : 0.0f;
    __syncthreads();
    for (int stride = 16/2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            s[threadIdx.x] += s[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[blockIdx.x] = s[0];
}
