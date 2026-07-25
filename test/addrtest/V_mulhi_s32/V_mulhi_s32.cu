extern "C" __global__ void V_mulhi_s32(const int* a, const int* b, int* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        out[i] = (int)(((long long)a[i] * (long long)b[i]) >> 32);
}
