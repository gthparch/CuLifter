extern "C" __global__ void N3_stencil_2d_cross(const float* in, float* out, int W, int H) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= 1 && col < W-1 && row >= 1 && row < H-1) {
        int c = row*W + col;
        out[c] = in[c-W] + in[c+W] + in[c-1] + in[c+1] + in[c];
    }
}
