extern "C" __global__ void D2_idx_unsigned(const float* in, float* out, int W, int H) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < W && row >= 1 && row < H - 1) {
        unsigned c = (unsigned)row * (unsigned)W + (unsigned)col;
        unsigned n = (unsigned)(row-1) * (unsigned)W + (unsigned)col;
        unsigned s = (unsigned)(row+1) * (unsigned)W + (unsigned)col;
        out[c] = in[n] + in[c] + in[s];
    }
}
