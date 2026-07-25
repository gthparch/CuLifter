extern "C" __global__ void addr_H4_pitch_2d_int(const float* in, float* out,
                                    int pitch, int W, int H) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < W && row < H) {
        int idx = (int)row * pitch + (int)col;
        out[idx] = in[idx];
    }
}
