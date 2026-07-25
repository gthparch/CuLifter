extern "C" __global__ void X1m_double(const double* in, double* out,
                                   size_t pitch, int W, int H) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= 1 && col < W - 1 && row >= 1 && row < H - 1) {
        double v_n = in[(size_t)(row - 1) * pitch + (size_t)col];
        double v_w = in[(size_t)row * pitch + (size_t)(col - 1)];
        double v_s = in[(size_t)(row + 1) * pitch + (size_t)col];
        double v_e = in[(size_t)row * pitch + (size_t)(col + 1)];
        double v_c = in[(size_t)row * pitch + (size_t)col];
        size_t oc = (size_t)row * pitch + (size_t)col;
        out[oc] = v_n + v_w + v_s + v_e + v_c;
    }
}
