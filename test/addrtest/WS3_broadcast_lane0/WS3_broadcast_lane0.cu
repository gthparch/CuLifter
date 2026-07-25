extern "C" __global__ void WS3_broadcast_lane0(const float* in, float* out, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (tid < N) ? in[tid] : 0.0f;
    float broadcast = __shfl_sync(0xffffffff, val, 0);
    if (tid < N) out[tid] = broadcast;
}
