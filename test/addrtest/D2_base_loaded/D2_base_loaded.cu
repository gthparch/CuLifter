extern "C" __global__ void D2_base_loaded(const float* const* ptrs, float* out, int W, int H, int which) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < W && row >= 1 && row < H - 1) {
        const float* in = ptrs[which];
        int c = row * W + col;
        out[c] = in[c - W] + in[c] + in[c + W];
    }
}
