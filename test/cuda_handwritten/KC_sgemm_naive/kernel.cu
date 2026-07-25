// KC_sgemm_naive — naive triple-loop SGEMM in column-major layout.
//
// Mirrors CB_sgemm_nn (M=N=K matched to the A100 capture's small case)
// but with NO cuBLAS dispatch:
//   - Single kernel, hand-written
//   - Single struct param, natural alignment
//   - No workspace, no tensor descriptor
//
// Compute: C[i,j] = alpha * sum_k A[i,k] * B[k,j] + beta * C[i,j]
// Layout:  A is M×K col-major (lda=M); B is K×N col-major (ldb=K);
//          C is M×N col-major (ldc=M).
//
// Grid: (M/BM, N/BN, 1)  Block: (BM, BN, 1)  — one thread per C element.
// For BM=BN=8 and M=N=K=64 → grid=(8,8,1), block=(8,8,1), 64 threads/block.

extern "C" {

struct KCSgemmParams {
    float* C;           // out, MxN col-major, lda=M
    const float* A;     // in,  MxK col-major, lda=M
    const float* B;     // in,  KxN col-major, lda=K
    float alpha;
    float beta;
    int M;
    int N;
    int K;
};

__global__ void kc_sgemm_naive(KCSgemmParams p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // row 0..M-1
    int j = blockIdx.y * blockDim.y + threadIdx.y;   // col 0..N-1
    if (i >= p.M || j >= p.N) return;

    float acc = 0.0f;
    for (int k = 0; k < p.K; ++k) {
        acc += p.A[i + k * p.M] * p.B[k + j * p.K];
    }
    int idx = i + j * p.M;
    p.C[idx] = p.alpha * acc + p.beta * p.C[idx];
}

}  // extern "C"
