extern "C" __global__ void D2_idx_sizet(const float* in, float* out, int W, int H) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < W && row >= 1 && row < H - 1) {
        size_t c = (size_t)row * (size_t)W + (size_t)col;
        size_t n = (size_t)(row-1) * (size_t)W + (size_t)col;
        size_t s = (size_t)(row+1) * (size_t)W + (size_t)col;
        out[c] = in[n] + in[c] + in[s];
    }
}
