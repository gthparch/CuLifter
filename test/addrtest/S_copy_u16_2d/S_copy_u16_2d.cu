extern "C" __global__ void S_copy_u16_2d(const unsigned short* in, unsigned short* out, int W, int H) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < W && row < H) {
        int c = row * W + col;
        out[c] = in[c];
    }
}
