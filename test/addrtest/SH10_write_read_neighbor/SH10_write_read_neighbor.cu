extern "C" __global__ void SH10_write_read_neighbor(const float* in, float* out, int N) {
    __shared__ float s[256];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) s[threadIdx.x] = in[i];
    __syncthreads();
    if (i < N) {
        int nb = threadIdx.x ^ 1;
        int gi = blockIdx.x * blockDim.x + nb;
        if (gi < N) out[i] = s[nb];
    }
}
