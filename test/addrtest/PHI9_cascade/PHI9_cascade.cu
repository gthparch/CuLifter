extern "C" __global__ void PHI9_cascade(const int* in, int* out, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    int val = in[tid];
    int r;
    // First switch
    switch (val % 5) {
        case 0: r = 10; break;
        case 1: r = 20; break;
        case 2: r = 30; break;
        case 3: r = 40; break;
        default: r = 50; break;
    }
    // PHI merges 5 paths into r
    // Second switch using r
    int s;
    switch (r) {
        case 10: s = val + r; break;
        case 20: s = val - r; break;
        case 30: s = val * 2; break;
        case 40: s = val / 2; break;
        default: s = val ^ r; break;
    }
    // PHI merges 5 more paths into s
    out[tid] = s + tid;
}
