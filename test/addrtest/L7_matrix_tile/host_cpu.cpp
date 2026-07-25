#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void L7_matrix_tile();

// MatrixDesc is passed by value in CUDA, so SASS sees flattened fields:
//   const float* data, int rows, int cols, int pitch
// followed by remaining kernel params: float* out, int N

int main() {
    constexpr int N = 1024, ROWS = 32, COLS = 32, PITCH = 32;
    constexpr int blockDim = 256;
    constexpr int gridDim = (N + blockDim - 1) / blockDim;

    float *in = new float[N];
    float *out = new float[N];
    std::memset(out, 0, N * sizeof(float));
    for (int i = 0; i < N; i++) in[i] = (float)(i % 7 + 1) * 0.1f;

    // Pass struct fields flattened: data, rows, cols, pitch, then out, N
    launchKernel(L7_matrix_tile, gridDim, blockDim,
                 (const float*)in, ROWS, COLS, PITCH, out, N);

    bool ok = true;
    for (int i = 0; i < N; i++) {
        int row = i / COLS, col = i % COLS;
        if (row < ROWS) {
            float expected = in[row * PITCH + col];
            if (std::fabs(out[i] - expected) > 1e-5f) {
                std::cout << "FAIL at " << i << ": got " << out[i]
                          << " expected " << expected << "\n";
                ok = false;
                break;
            }
        }
    }
    std::cout << (ok ? "TEST PASSED" : "FAIL") << std::endl;

    delete[] in; delete[] out;
    return ok ? 0 : 1;
}
