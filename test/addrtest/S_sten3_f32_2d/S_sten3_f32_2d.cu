extern "C" __global__ void S_sten3_f32_2d(const float* in, float* out, int W, int H) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < W && row >= 1 && row < H - 1) {
        int c = row * W + col;
        out[c] = (float)(in[c - W] + in[c] + in[c + W]);
    }
}
