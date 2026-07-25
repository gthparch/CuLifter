// KC_sgemm_tiled_smem — tiled SGEMM with shared-memory tile loader.
// 16x16 tiles, one thread per output element, classic textbook impl.
// Exercises lifter handling of __shared__ memory.

extern "C" {

struct KCSgemmTiledParams {
    float* C;           // out, MxN col-major, lda=M
    const float* A;     // in,  MxK col-major, lda=M
    const float* B;     // in,  KxN col-major, lda=K
    float alpha;
    float beta;
    int M;
    int N;
    int K;
};

#define TILE 16

__global__ void kc_sgemm_tiled(KCSgemmTiledParams p) {
    __shared__ float Asub[TILE][TILE];
    __shared__ float Bsub[TILE][TILE];

    int i = blockIdx.x * TILE + threadIdx.x;        // row
    int j = blockIdx.y * TILE + threadIdx.y;        // col
    int tx = threadIdx.x, ty = threadIdx.y;

    float acc = 0.0f;
    int n_tiles = (p.K + TILE - 1) / TILE;

    for (int t = 0; t < n_tiles; ++t) {
        int aRow = i;
        int aCol = t * TILE + ty;
        int bRow = t * TILE + tx;
        int bCol = j;
        Asub[tx][ty] = (aRow < p.M && aCol < p.K) ? p.A[aRow + aCol * p.M] : 0.0f;
        Bsub[tx][ty] = (bRow < p.K && bCol < p.N) ? p.B[bRow + bCol * p.K] : 0.0f;
        __syncthreads();
        for (int k = 0; k < TILE; ++k) acc += Asub[tx][k] * Bsub[k][ty];
        __syncthreads();
    }
    if (i < p.M && j < p.N) {
        int idx = i + j * p.M;
        p.C[idx] = p.alpha * acc + p.beta * p.C[idx];
    }
}

}  // extern "C"
