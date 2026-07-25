// Variants: full f64 division and rsqrt (MUFU.RSQ64H path).
#include <cmath>
extern "C" __global__ void f64_div(const double* a, const double* b, double* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i] / b[i];
}
extern "C" __global__ void f64_rsqrt(const double* x, double* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = rsqrt(x[i]);
}
