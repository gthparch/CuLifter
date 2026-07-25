extern "C" __global__ void SH13_2d_tile_8x8(const float* in, float* out, int W, int H) {
    __shared__ float tile[8][8];
    int col = blockIdx.x * 8 + threadIdx.x;
    int row = blockIdx.y * 8 + threadIdx.y;
    if (col < W && row < H)
        tile[threadIdx.y][threadIdx.x] = in[row * W + col];
    __syncthreads();
    if (col < W && row < H)
        out[row * W + col] = tile[threadIdx.y][threadIdx.x];
}
