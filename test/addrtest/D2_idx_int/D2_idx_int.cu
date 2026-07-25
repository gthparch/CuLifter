extern "C" __global__ void D2_idx_int(const float* in, float* out, int W, int H) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < W && row >= 1 && row < H - 1) {
        int c = (int)row * (int)W + (int)col;
        int n = (int)(row-1) * (int)W + (int)col;
        int s = (int)(row+1) * (int)W + (int)col;
        out[c] = in[n] + in[c] + in[s];
    }
}
