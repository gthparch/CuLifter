#include <cuda_runtime.h>

// Kernel: C[i] = A[i] / B[i]
extern "C" __global__ void vector_div(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float div1 = A[i] / B[i];
        // float div2 = float(int(A[i]) / int(B[i]));
        C[i] = div1;
    }
}
