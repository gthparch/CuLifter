extern "C" __global__ void D2_sign_neg(const float* in, float* out, int W, int H) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < W && row >= 2) {
        int c = row * W + col;
        out[c] = in[c - 2*W] + in[c - W] + in[c];
    }
}
