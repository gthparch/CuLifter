extern "C" __global__ void X1m_pitch_int(const float* in, float* out,
                                   int pitch, int W, int H) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= 1 && col < W - 1 && row >= 1 && row < H - 1) {
        float v_n = in[(int)(row - 1) * pitch + (int)col];
        float v_w = in[(int)row * pitch + (int)(col - 1)];
        float v_s = in[(int)(row + 1) * pitch + (int)col];
        float v_e = in[(int)row * pitch + (int)(col + 1)];
        float v_c = in[(int)row * pitch + (int)col];
        int oc = (int)row * pitch + (int)col;
        out[oc] = v_n + v_w + v_s + v_e + v_c;
    }
}
