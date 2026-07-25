extern "C" __global__ void D2_ctrl_clamp(const float* in, float* out, int W, int H) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < W && row < H) {
        int c = row * W + col;
        int nr = (row > 0) ? row - 1 : 0;
        int sr = (row < H - 1) ? row + 1 : H - 1;
        out[c] = in[nr * W + col] + in[c] + in[sr * W + col];
    }
}
