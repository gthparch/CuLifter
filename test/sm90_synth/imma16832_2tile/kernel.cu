// IMMA.16832 multi-tile int8 GEMM fixture (one warp, 32x16x64 = 8 atoms).
//
// The MULTI-TILE companion to imma16832. A single m16n8k32 atom already exercises
// the M16 row split (m=g vs g+8) and the K32 split (k<16 vs >=16), but a single
// atom's C/D store+load+map are self-consistent, so it cannot expose a transposed
// m<->n C/D convention. Only a GEMM with OFF-DIAGONAL output tiles does (the
// DMMA.8x8x4 / IMMA.8816 fixes learned this the hard way). Here one warp computes
//   D[32][16] = C[32][16] + sum_{k<64} A[32][k] * B[16][k]   (A row-major, B = N x K)
// as 2x2 output tiles of 16x8 (M=2 tiles of 16, N=2 tiles of 8), each accumulating
// 2 IMMA.16832 atoms over K (=64).  2*2*2 = 8 atoms.
//
// Per atom (output tile (mi,ni), k-group ki), each lane uses the cute
// MMA_Traits<SM80_16x8x32_S32S8S8S32_TN> map (g=lane/4, tg=lane%4):
//   A: 4 regs, each 4 packed s8 (byte b):
//     r0: A[mi*16 + g  ][ki*32 +      tg*4+b]   r1: A[mi*16 + g+8][ki*32 +      tg*4+b]
//     r2: A[mi*16 + g  ][ki*32 + 16 + tg*4+b]   r3: A[mi*16 + g+8][ki*32 + 16 + tg*4+b]
//   B: 2 regs, each 4 packed s8:
//     r0: B[ni*8 + g][ki*32 +      tg*4+b]      r1: B[ni*8 + g][ki*32 + 16 + tg*4+b]
//   C/D: 4 int32 regs (accumulator within the (mi,ni) tile):
//     v0: tile[g  ][tg*2+0]  v1: tile[g  ][tg*2+1]
//     v2: tile[g+8][tg*2+0]  v3: tile[g+8][tg*2+1]
extern "C" __global__ void kern_imma16832_2tile(const signed char* A, const signed char* B,
                                                const int* C, int* D)
{
    const int lane = threadIdx.x & 31;
    const int g  = lane >> 2;   // 0..7
    const int tg = lane & 3;    // 0..3
    constexpr int LDA = 64, LDB = 64, LDD = 16;  // K, K, N

    for (int mi = 0; mi < 2; ++mi) {
        for (int ni = 0; ni < 2; ++ni) {
            // seed accumulator from C[mi*16 + {g,g+8}][ni*8 + tg*2 + {0,1}]
            int d0 = C[(mi * 16 + g    ) * LDD + (ni * 8 + tg * 2 + 0)];
            int d1 = C[(mi * 16 + g    ) * LDD + (ni * 8 + tg * 2 + 1)];
            int d2 = C[(mi * 16 + g + 8) * LDD + (ni * 8 + tg * 2 + 0)];
            int d3 = C[(mi * 16 + g + 8) * LDD + (ni * 8 + tg * 2 + 1)];
            for (int ki = 0; ki < 2; ++ki) {
                unsigned int a0 = 0, a1 = 0, a2 = 0, a3 = 0, b0 = 0, b1 = 0;
#pragma unroll
                for (int b = 0; b < 4; ++b) {
                    a0 |= (static_cast<unsigned int>(static_cast<unsigned char>(A[(mi*16 + g    )*LDA + (ki*32 +      tg*4 + b)]))) << (8*b);
                    a1 |= (static_cast<unsigned int>(static_cast<unsigned char>(A[(mi*16 + g + 8)*LDA + (ki*32 +      tg*4 + b)]))) << (8*b);
                    a2 |= (static_cast<unsigned int>(static_cast<unsigned char>(A[(mi*16 + g    )*LDA + (ki*32 + 16 + tg*4 + b)]))) << (8*b);
                    a3 |= (static_cast<unsigned int>(static_cast<unsigned char>(A[(mi*16 + g + 8)*LDA + (ki*32 + 16 + tg*4 + b)]))) << (8*b);
                    b0 |= (static_cast<unsigned int>(static_cast<unsigned char>(B[(ni*8  + g    )*LDB + (ki*32 +      tg*4 + b)]))) << (8*b);
                    b1 |= (static_cast<unsigned int>(static_cast<unsigned char>(B[(ni*8  + g    )*LDB + (ki*32 + 16 + tg*4 + b)]))) << (8*b);
                }
                asm volatile(
                    "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 "
                    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                    : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                      "r"(b0), "r"(b1),
                      "r"(d0), "r"(d1), "r"(d2), "r"(d3));
            }
            D[(mi * 16 + g    ) * LDD + (ni * 8 + tg * 2 + 0)] = d0;
            D[(mi * 16 + g    ) * LDD + (ni * 8 + tg * 2 + 1)] = d1;
            D[(mi * 16 + g + 8) * LDD + (ni * 8 + tg * 2 + 0)] = d2;
            D[(mi * 16 + g + 8) * LDD + (ni * 8 + tg * 2 + 1)] = d3;
        }
    }
}
