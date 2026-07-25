extern "C" __global__ void PHI4_loop_break(const int* in, int* out, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    int val = in[tid];
    int acc = 0;
    for (int j = 0; j < 16; j++) {
        acc += j;
        if (acc > val && val > 0) break;
    }
    out[tid] = acc;
}
