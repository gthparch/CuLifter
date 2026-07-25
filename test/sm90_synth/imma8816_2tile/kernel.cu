// IMMA.8816 multi-tile int8 GEMM fixture (one warp, 16x16x32 = 4 atoms x 2 K-steps).
//
// This is the MULTI-TILE companion to imma8816: a single 8x8 atom cannot
// distinguish the C/D m<->n fragment convention (store+load+map are
// self-consistent), so it silently hides a transposed C/D layout. Only a GEMM
// with OFF-DIAGONAL output tiles exposes it (the DMMA.8x8x4 fix learned this the
// hard way on 19_tensorop_canonical). Here one warp computes a full
//   D[16][16] = C[16][16] + sum_{k<32} A[16][k] * B[16][k]   (A row-major, B = N x K)
// as 2x2 output tiles of 8x8, each accumulating 2 IMMA.8816 atoms over K (=32).
//
// Per atom (output tile (mi,ni), k-group ki), each lane uses the cute
// MMA_Traits<SM75_8x8x16_S32S8S8S32_TN> map (g=lane/4, tg=lane%4):
//   a[v] = A[mi*8 + g][ki*16 + tg*4 + v]   v in 0..3   (packed 4xs8, little-endian)
//   b[v] = B[ni*8 + g][ki*16 + tg*4 + v]   v in 0..3
//   c/d[v] = tile[m=g][n=tg*2 + v]         v in 0..1   (int32 accumulator)
extern "C" __global__ void kern_imma8816_2tile(const signed char* A, const signed char* B,
                                               const int* C, int* D)
{
    const int lane = threadIdx.x & 31;
    const int g  = lane >> 2;   // 0..7
    const int tg = lane & 3;    // 0..3
    constexpr int LDA = 32, LDB = 32, LDD = 16;  // K, K, N

    for (int mi = 0; mi < 2; ++mi) {
        for (int ni = 0; ni < 2; ++ni) {
            // seed accumulator from C[mi*8+g][ni*8 + tg*2 + v]
            int d0 = C[(mi * 8 + g) * LDD + (ni * 8 + tg * 2 + 0)];
            int d1 = C[(mi * 8 + g) * LDD + (ni * 8 + tg * 2 + 1)];
            for (int ki = 0; ki < 2; ++ki) {
                unsigned int ar = 0, br = 0;
#pragma unroll
                for (int v = 0; v < 4; ++v) {
                    ar |= (static_cast<unsigned int>(static_cast<unsigned char>(
                        A[(mi * 8 + g) * LDA + (ki * 16 + tg * 4 + v)]))) << (8 * v);
                    br |= (static_cast<unsigned int>(static_cast<unsigned char>(
                        B[(ni * 8 + g) * LDB + (ki * 16 + tg * 4 + v)]))) << (8 * v);
                }
                asm volatile(
                    "mma.sync.aligned.m8n8k16.row.col.satfinite.s32.s8.s8.s32 "
                    "{%0, %1}, {%2}, {%3}, {%4, %5};\n"
                    : "=r"(d0), "=r"(d1)
                    : "r"(ar), "r"(br), "r"(d0), "r"(d1));
            }
            D[(mi * 8 + g) * LDD + (ni * 8 + tg * 2 + 0)] = d0;
            D[(mi * 8 + g) * LDD + (ni * 8 + tg * 2 + 1)] = d1;
        }
    }
}
