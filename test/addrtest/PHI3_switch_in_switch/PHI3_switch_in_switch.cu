extern "C" __global__ void PHI3_switch_in_switch(const int* in, int* out, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    int val = in[tid];
    int r = 0;
    switch (val % 3) {
        case 0:
            switch ((val / 3) % 4) {
                case 0: r = val + 1; break;
                case 1: r = val + 2; break;
                case 2: r = val + 4; break;
                default: r = val + 8; break;
            }
            break;
        case 1:
            r = val * 2;
            break;
        default:
            switch ((val / 3) % 3) {
                case 0: r = val - 1; break;
                case 1: r = val - 2; break;
                default: r = val - 4; break;
            }
            break;
    }
    out[tid] = r;
}
