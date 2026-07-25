extern "C" __global__ void IM_shl_17(const int* in, int* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        int x = in[i];
        out[i] = x << 17;
    }
}
