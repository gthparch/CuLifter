extern "C" __global__ void X1m_pos_only(const float* in, float* out,
                                   size_t pitch, int W, int H) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= 1 && col < W - 1 && row >= 1 && row < H - 1) {
        float v_s = in[(size_t)(row + 1) * pitch + (size_t)col];
        float v_e = in[(size_t)row * pitch + (size_t)(col + 1)];
        float v_c = in[(size_t)row * pitch + (size_t)col];
        size_t oc = (size_t)row * pitch + (size_t)col;
        out[oc] = v_s + v_e + v_c;
    }
}
