extern "C" __global__ void V_mulhi_mixed(const int* in, int* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        out[i] = (int)(((long long)in[i] * 0x55555556LL) >> 32);
}
