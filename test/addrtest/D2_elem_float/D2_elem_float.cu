extern "C" __global__ void D2_elem_float(const float* in, float* out, int W, int H) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < W && row >= 1 && row < H - 1) {
        int c = row * W + col;
        out[c] = in[c - W] + in[c] + in[c + W];
    }
}
