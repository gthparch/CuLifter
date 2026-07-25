extern "C" __global__ void PHI0_if_else(const int* in, int* out, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    int val = in[tid];
    int r;
    if (val > 0) r = val * 2;
    else r = val + 100;
    out[tid] = r + tid;
}
