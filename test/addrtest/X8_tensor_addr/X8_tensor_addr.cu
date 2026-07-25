extern "C" __global__ void X8_tensor_addr(const float* in, float* out,
    size_t C, size_t H_stride, size_t W_stride, int B, int Ht, int Wt, int Ch) {
    int b = blockIdx.z;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < B && h < Ht && w < Wt) {
        for (int c = 0; c < Ch; c++) {
            size_t idx = (size_t)b * H_stride + (size_t)h * W_stride + (size_t)w * C + (size_t)c;
            size_t oidx = ((size_t)b * Ht * Wt + (size_t)h * Wt + (size_t)w) * Ch + c;
            out[oidx] = in[idx];
        }
    }
}
