// tma_2d_tile: 2D tile-based access pattern
// Kernel: out[row][col] = in[row][col] * 2.0f
// Tests descriptor addressing with stride computation

extern "C" __global__
void tma_2d_tile(const float* __restrict__ in, float* __restrict__ out,
                 int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < rows && col < cols) {
        int idx = row * cols + col;
        out[idx] = in[idx] * 2.0f;
    }
}
