extern "C" __global__ void D2_ctrl_ternary(const float* in, float* out, int W, int H) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < W && row < H) {
        int c = row * W + col;
        float north = (row > 0) ? in[c - W] : 0.0f;
        float south = (row < H - 1) ? in[c + W] : 0.0f;
        out[c] = north + in[c] + south;
    }
}
