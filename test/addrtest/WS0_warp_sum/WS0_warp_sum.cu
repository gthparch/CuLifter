extern "C" __global__ void WS0_warp_sum(const float* in, float* out, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (tid < N) ? in[tid] : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    if (threadIdx.x % 32 == 0 && tid < N)
        out[tid / 32] = val;
}
