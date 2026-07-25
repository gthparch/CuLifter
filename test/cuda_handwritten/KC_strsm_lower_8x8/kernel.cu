// KC_strsm_lower_8x8 — 8x8 lower-triangular triangular solve, single block.
// Solves L * X = alpha * B  for X (in-place: X overwrites B).
//
// L is 8x8 col-major lower-triangular (lda=8). X and B are 8x8 col-major.
// One thread per column of X.  Within each column, forward substitution.
//
// Shadows CB_strsm but with NO internal allocation, NO multi-block, just the
// canonical algorithm.

extern "C" {

struct KCTrsmParams {
    const float* L;     // 8x8 lower triangular, lda=8
    float* X;           // 8x8 in/out (initially holds B; ends holding X)
    float alpha;
};

__global__ void kc_strsm_lower_8x8(KCTrsmParams p) {
    int j = threadIdx.x;        // column of X (0..7), one thread per column
    if (j >= 8) return;
    // Forward substitution within column j
    for (int i = 0; i < 8; ++i) {
        float sum = p.alpha * p.X[i + j*8];
        for (int k = 0; k < i; ++k) sum -= p.L[i + k*8] * p.X[k + j*8];
        p.X[i + j*8] = sum / p.L[i + i*8];
    }
}

}  // extern "C"
