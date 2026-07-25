extern "C" __global__ void PHI7_diamond(const int* in, int* out, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    int val = in[tid];
    int r;
    if (val > 0) {
        if (val > 50) r = val * 2;    // path A
        else r = val + 10;            // path B
    } else {
        if (val < -50) r = val * val;  // path C
        else r = val + 100;            // path D
    }
    // PHI merges paths A, B, C, D
    int s;
    if (r > 200) s = r / 2;
    else s = r * 2;
    out[tid] = s;
}
