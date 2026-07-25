extern "C" __global__ void addr_X1_stencil_64pitch_float(const float* in, float* out,
                                    size_t pitch, int W, int H) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= 1 && col < W - 1 && row >= 1 && row < H - 1) {
        size_t c = (size_t)row * pitch + (size_t)col;
        size_t n = (size_t)(row - 1) * pitch + (size_t)col;
        size_t s = (size_t)(row + 1) * pitch + (size_t)col;
        size_t w = (size_t)row * pitch + (size_t)(col - 1);
        size_t e = (size_t)row * pitch + (size_t)(col + 1);
        out[c] = in[n] + in[s] + in[w] + in[e] + in[c];
    }
}
