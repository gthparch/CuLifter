extern "C" __global__ void PHI1_switch5(const int* in, int* out, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    int val = in[tid];
    int r;
    switch (val % 5) {
        case 0: r = val * 2; break;
        case 1: r = val + 100; break;
        case 2: r = val - 50; break;
        case 3: r = val * val; break;
        default: r = -val; break;
    }
    out[tid] = r;
}
