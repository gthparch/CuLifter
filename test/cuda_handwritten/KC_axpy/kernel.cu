// KC_axpy — trivial single-pass axpy for sanity-checking the shadow-kernel
// pipeline (struct param + ARG_BASE 0x160 + launchKernel + cpu_ref compare).
//
// Compute: y[i] = alpha * x[i] + y[i]   for i in [0, N)

extern "C" {

struct KCAxpyParams {
    float* y;           // out (in-place)
    const float* x;
    float alpha;
    int n;
};

__global__ void kc_axpy(KCAxpyParams p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < p.n) {
        p.y[i] = p.alpha * p.x[i] + p.y[i];
    }
}

}  // extern "C"
