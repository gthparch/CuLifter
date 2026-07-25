extern "C" __global__ void X5_gather_u64(const float* data,
    const unsigned long long* indices, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        out[i] = data[(int)(indices[i] % (unsigned long long)N)];
}
