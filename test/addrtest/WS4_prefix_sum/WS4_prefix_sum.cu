extern "C" __global__ void WS4_prefix_sum(const float* in, float* out, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (tid < N) ? in[tid] : 0.0f;
    for (int d = 1; d < 32; d <<= 1) {
        float n = __shfl_up_sync(0xffffffff, val, d);
        if (threadIdx.x % 32 >= d) val += n;
    }
    if (tid < N) out[tid] = val;
}
