extern "C" __global__ void H4m_idx_ull(const float* in, float* out,
                                   unsigned long long pitch, int W, int H) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < W && row < H) {
        unsigned long long idx = (unsigned long long)row * pitch + (unsigned long long)col;
        out[idx] = in[idx];
    }
}
