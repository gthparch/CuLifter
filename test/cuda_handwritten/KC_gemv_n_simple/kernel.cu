// KC_gemv_n_simple — y(M) = alpha * A(MxN) * x(N) + beta * y(M)
// Column-major A, lda = M.  One thread per output row.
// No tensor descriptor (cuBLAS style) — just plain ptrs.

extern "C" {

struct KCGemvParams {
    const float* A;     // MxN col-major, lda=M
    const float* x;     // N
    float* y;           // M (in/out)
    float alpha;
    float beta;
    int M;
    int N;
};

__global__ void kc_gemv_n_simple(KCGemvParams p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;     // row
    if (i >= p.M) return;
    float acc = 0.0f;
    for (int j = 0; j < p.N; ++j) acc += p.A[i + j * p.M] * p.x[j];
    p.y[i] = p.alpha * acc + p.beta * p.y[i];
}

}  // extern "C"
