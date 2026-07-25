#include <cuda_runtime.h>

// Kernel: y[i] = a * x[i] + y[i]
// Matches symbol: _Z5saxpyifPfS_
__global__ void saxpy(int n, float a, const float* x, float* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}

