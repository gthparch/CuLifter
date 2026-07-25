extern "C" __global__ void SH2_neighbor_shared(const float* in, float* out, int N) {
    __shared__ float tile[256];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        tile[threadIdx.x] = in[i];
    __syncthreads();
    if (i >= 1 && i < N - 1 && threadIdx.x >= 1 && threadIdx.x < blockDim.x - 1)
        out[i] = tile[threadIdx.x - 1] + tile[threadIdx.x] + tile[threadIdx.x + 1];
}
