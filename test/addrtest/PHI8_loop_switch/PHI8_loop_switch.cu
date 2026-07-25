extern "C" __global__ void PHI8_loop_switch(const int* in, int* out, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    int val = in[tid];
    int acc = val;
    for (int j = 0; j < 4; j++) {
        switch (acc % 3) {
            case 0: acc = acc + j + 1; break;
            case 1: acc = acc - j; break;
            default: acc = acc ^ (j + 1); break;
        }
    }
    out[tid] = acc;
}
