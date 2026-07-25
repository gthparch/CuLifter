extern "C" __global__ void V_idiv_10(const int* in, int* out_q, int* out_r, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        out_q[i] = in[i] / 10;
        out_r[i] = in[i] % 10;
    }
}
