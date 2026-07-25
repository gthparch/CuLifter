extern "C" __global__ void S_sten5_u16_2d(const unsigned short* in, unsigned short* out, int W, int H) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < W && row >= 2 && row < H - 2) {
        int c = row * W + col;
        out[c] = (unsigned short)(in[c - 2*W] + in[c - W] + in[c] + in[c + W] + in[c + 2*W]);
    }
}
