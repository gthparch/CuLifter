extern "C" __global__ void X6_scatter_u64(const float* data,
    const unsigned long long* indices, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        out[(int)(indices[i] % (unsigned long long)N)] = data[i];
}
