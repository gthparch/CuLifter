extern "C" __global__ void IM_and_signbit(const int* in, int* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        int x = in[i];
        out[i] = (int)((unsigned)x & 0x80000000u);
    }
}
