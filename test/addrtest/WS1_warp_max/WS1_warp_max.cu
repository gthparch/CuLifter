extern "C" __global__ void WS1_warp_max(const float* in, float* out, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (tid < N) ? in[tid] : -1e30f;
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    if (threadIdx.x % 32 == 0 && tid < N) out[tid / 32] = val;
}
