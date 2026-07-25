extern "C" __global__ void PHI5_three_level(const int* in, int* out, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    int val = in[tid];
    int r = 0;
    if (val > 0) {
        switch (val % 4) {
            case 0: r = val; break;
            case 1:
                if (val > 50) r = val * 2;
                else r = val + 20;
                break;
            case 2: r = val - 10; break;
            default: r = val / 2; break;
        }
    } else {
        switch ((-val) % 3) {
            case 0: r = -val; break;
            case 1:
                if (val < -50) r = val * val;
                else r = val + 50;
                break;
            default: r = 42; break;
        }
    }
    out[tid] = r + tid;
}
