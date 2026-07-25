extern "C" __global__ void SH1_tiled_copy(const float* in, float* out, int N) {
    __shared__ float tile[256];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        tile[threadIdx.x] = in[i];
    __syncthreads();
    if (i < N)
        out[i] = tile[threadIdx.x];
}
