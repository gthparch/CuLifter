struct TensorMeta { const float* ptr; size_t pitch; int W; int H; };

extern "C" __global__ void X3_struct_ptr_pitch(TensorMeta meta, float* out) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < meta.W && row < meta.H) {
        size_t idx = (size_t)row * meta.pitch + (size_t)col;
        out[row * meta.W + col] = meta.ptr[idx];
    }
}
