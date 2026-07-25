extern "C" __global__ void S_flat2d_u16(const unsigned short* in, unsigned short* out, int W, int H, int pitch) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < W && row < H) {
        int src = row * pitch + col;
        int dst = row * W + col;
        out[dst] = in[src];
    }
}
