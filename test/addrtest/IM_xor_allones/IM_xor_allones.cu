extern "C" __global__ void IM_xor_allones(const int* in, int* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        int x = in[i];
        out[i] = x ^ 0xFFFFFFFF;
    }
}
