struct MatrixDesc { const float* data; int rows; int cols; int pitch; };

extern "C" __global__ void L7_matrix_tile(MatrixDesc src, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        int row = i / src.cols;
        int col = i % src.cols;
        if (row < src.rows)
            out[i] = src.data[row * src.pitch + col];
    }
}
