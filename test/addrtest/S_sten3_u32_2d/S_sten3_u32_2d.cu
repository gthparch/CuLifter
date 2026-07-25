extern "C" __global__ void S_sten3_u32_2d(const unsigned int* in, unsigned int* out, int W, int H) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < W && row >= 1 && row < H - 1) {
        int c = row * W + col;
        out[c] = (unsigned int)(in[c - W] + in[c] + in[c + W]);
    }
}
