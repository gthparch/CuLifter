extern "C" __global__ void addr_H4_pitch_2d_sizet(const float* in, float* out,
                                    size_t pitch, int W, int H) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < W && row < H) {
        size_t idx = (size_t)row * pitch + (size_t)col;
        out[idx] = in[idx];
    }
}
