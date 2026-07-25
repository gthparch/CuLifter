extern "C" __global__ void V_mulhi_u32(const unsigned int* a, const unsigned int* b,
    unsigned int* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        out[i] = (unsigned int)(((unsigned long long)a[i] * (unsigned long long)b[i]) >> 32);
}
