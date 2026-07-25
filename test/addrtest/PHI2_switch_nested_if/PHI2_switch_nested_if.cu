extern "C" __global__ void PHI2_switch_nested_if(const int* in, int* out, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    int val = in[tid];
    int r;
    switch (val % 4) {
        case 0:
            if (val > 50) r = val * 3;
            else r = val + 10;
            break;
        case 1:
            if (val < -20) r = val * val;
            else r = val - 5;
            break;
        case 2:
            r = (val > 0) ? val : -val;
            break;
        default:
            r = val ^ 0xFF;
            break;
    }
    out[tid] = r + tid;
}
