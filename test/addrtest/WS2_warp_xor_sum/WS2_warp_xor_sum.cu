extern "C" __global__ void WS2_warp_xor_sum(const float* in, float* out, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (tid < N) ? in[tid] : 0.0f;
    for (int mask = 16; mask > 0; mask >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, mask);
    if (tid < N) out[tid] = val;
}
