extern "C" __global__ void X1m_radius2(const float* in, float* out,
                                   size_t pitch, int W, int H) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= 2 && col < W - 2 && row >= 2 && row < H - 2) {
        float v_n = in[(size_t)(row - 2) * pitch + (size_t)col];
        float v_w = in[(size_t)row * pitch + (size_t)(col - 2)];
        float v_s = in[(size_t)(row + 2) * pitch + (size_t)col];
        float v_e = in[(size_t)row * pitch + (size_t)(col + 2)];
        float v_c = in[(size_t)row * pitch + (size_t)col];
        size_t oc = (size_t)row * pitch + (size_t)col;
        out[oc] = v_n + v_w + v_s + v_e + v_c;
    }
}
