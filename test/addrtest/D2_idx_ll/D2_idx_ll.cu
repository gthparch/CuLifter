extern "C" __global__ void D2_idx_ll(const float* in, float* out, int W, int H) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < W && row >= 1 && row < H - 1) {
        long long c = (long long)row * (long long)W + (long long)col;
        long long n = (long long)(row-1) * (long long)W + (long long)col;
        long long s = (long long)(row+1) * (long long)W + (long long)col;
        out[c] = in[n] + in[c] + in[s];
    }
}
