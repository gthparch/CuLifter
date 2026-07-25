extern "C" __global__ void D2_base_selected(const float* in1, const float* in2,
    float* out, int W, int H, int selector) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < W && row >= 1 && row < H - 1) {
        const float* in = (selector == 0) ? in1 : in2;
        int c = row * W + col;
        out[c] = in[c - W] + in[c] + in[c + W];
    }
}
