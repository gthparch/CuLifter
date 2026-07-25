extern "C" __global__ void SH3_shared_2d_tile(const float* in, float* out, int W, int H) {
    __shared__ float tile[16][16];
    int col = blockIdx.x * 16 + threadIdx.x;
    int row = blockIdx.y * 16 + threadIdx.y;
    if (col < W && row < H)
        tile[threadIdx.y][threadIdx.x] = in[row * W + col];
    __syncthreads();
    if (col < W && row < H)
        out[row * W + col] = tile[threadIdx.y][threadIdx.x];
}
