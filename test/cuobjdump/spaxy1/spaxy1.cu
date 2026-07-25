#include <cuda_runtime.h>

// Kernel: y[i] = x[i] + y[i]
// Matches symbol: _Z5saxpyiPfS_
__global__ void saxpy(int n, const float* x, float* y) {
    // Original SASS uses only threadIdx.x (single block expected)
    int i = threadIdx.x;
    if (i < n) {
        y[i] = x[i] + y[i];
    }
}

