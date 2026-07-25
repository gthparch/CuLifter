extern "C" __global__ void IM_and_ff(const int* in, int* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        int x = in[i];
        out[i] = x & 0xFF;
    }
}
