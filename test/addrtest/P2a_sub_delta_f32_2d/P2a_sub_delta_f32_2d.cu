extern "C" __global__ void P2a_sub_delta_f32_2d(const float* in, float* out, int W, int H, int row_delta) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < W && row >= row_delta && row < H) {
        int c = row * W + col;
        out[c] = in[(row - row_delta) * W + col];
    }
}
