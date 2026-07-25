#include <cstdio>
#include <cmath>

// A separate, non-inlined device function -> emitted as its own top-level
// SASS .text section reached by CALL.ABS.NOINC + R_CUDA_ABS55_16_34 reloc.
__noinline__ __device__ double helper(double x) {
    double y = x * 3.0 + 1.0;     // f64 arithmetic in the callee
    if (x < 0.0) y = -y;          // a branch so the callee has its own CFG
    return y;
}

__global__ void kernel(const double* __restrict__ in,
                       double* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = helper(in[i]);
}

int main() {
    const int n = 256;
    double *in, *out;
    cudaMallocManaged(&in, n * sizeof(double));
    cudaMallocManaged(&out, n * sizeof(double));
    for (int i = 0; i < n; i++) in[i] = (double)(i - 128) * 0.5;
    kernel<<<(n + 63) / 64, 64>>>(in, out, n);
    cudaDeviceSynchronize();
    int bad = 0;
    for (int i = 0; i < n; i++) {
        double x = in[i];
        double ref = x * 3.0 + 1.0;
        if (x < 0.0) ref = -ref;
        if (fabs(ref - out[i]) > 1e-9) {
            if (bad < 5) printf("MISMATCH i=%d x=%f ref=%f got=%f\n", i, x, ref, out[i]);
            bad++;
        }
    }
    printf("%s\n", bad == 0 ? "PASS" : "FAIL");
    return 0;
}
