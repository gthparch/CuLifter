extern "C" __global__ void D2_pitch64(const float* in, float* out, int pitch, int W, int H) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < W && row >= 1 && row < H - 1) {
        int c = row * pitch + col;
        out[c] = in[c - pitch] + in[c] + in[c + pitch];
    }
}
