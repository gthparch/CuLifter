#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

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

int main(int argc, char** argv) {
    constexpr int N = 1024, ROWS = 32, COLS = 32, PITCH = 32;
    float *h_in = new float[N];
    float h_out[N];
    memset(h_out, 0, sizeof(h_out));
    for (int i = 0; i < N; i++) h_in[i] = (float)(i % 7 + 1) * 0.1f;
    float *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, N * sizeof(float));
    MatrixDesc desc = {d_in, ROWS, COLS, PITCH};
    L7_matrix_tile<<<(N+255)/256, 256>>>(desc, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);
    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb");
        fwrite(h_out, sizeof(h_out[0]), sizeof(h_out)/sizeof(h_out[0]), f);
        fclose(f);
    }
    bool ok = true;
    for (int i = 0; i < N; i++) {
        int row = i / COLS, col = i % COLS;
        if (row < ROWS) {
            float expected = h_in[row * PITCH + col];
            if (fabsf(h_out[i] - expected) > 1e-5f) {
                printf("FAIL at %d\n", i); ok = false; break;
            }
        }
    }
    printf(ok ? "TEST PASSED\n" : "FAIL\n");
    cudaFree(d_in); cudaFree(d_out); delete[] h_in;
    return ok ? 0 : 1;
}
