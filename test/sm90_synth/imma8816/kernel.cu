// IMMA.8816 int8 tensor-core MMA fixture (one warp, single m8n8k16 atom).
//
// Each lane loads its A/B/C fragments from plain buffers using the CUTLASS cute
// MMA_Traits<SM75_8x8x16_S32S8S8S32_TN> mapping, performs ONE mma.sync.m8n8k16
// (s32 = s8 * s8 + s32, satfinite), and scatters its 2 D fragments back:
//   g = lane/4 (0..7) = row m; tg = lane%4 (0..3) = K-quad / N-pair selector.
//   A: 1 reg = 4 packed s8 -> a[v] = A[m=g][k=tg*4 + v]   (A is M8 x K16, row-major)
//   B: 1 reg = 4 packed s8 -> b[v] = B[n=g][k=tg*4 + v]   (B is N8 x K16, i.e. col-major K x N)
//   C/D: 2 s32 regs        -> c/d[v] = C[m=g][n=tg*2 + v] (v in {0,1}, C/D M8 x N8 int32)
//     (cute column-major (M-major) codomain — the SAME convention the DMMA.8x8x4
//      atom uses; a single 8x8 atom cannot distinguish the C/D m<->n convention,
//      but a multi-tile GEMM (08_turing_tensorop_gemm) does. Verified multi-tile.)
//   int8 byte order: byte0 (bits[7:0]) = v=0  (little-endian, matches __dp4a packing).
// Result: D[m][n] = C[m][n] + sum_{k<16} A[m][k]*B[n][k].
//
// This isolates the IMMA.8816 atom — the path 08_turing_tensorop_gemm (canonical int8
// tensorop GEMM, 256 IMMA.8816.S8.S8.SAT) exercises and that the lifter previously
// stubbed as a C->D identity (the A*B product was dropped on x86).
extern "C" __global__ void kern_imma8816(const signed char* A, const signed char* B,
                                         const int* C, int* D)
{
    const int lane = threadIdx.x & 31;
    const int g  = lane >> 2;   // lane/4 in {0..7}  -> row m
    const int tg = lane & 3;    // lane%4 in {0..3}  -> K-quad / N-pair

    // Pack this lane's 4 A bytes and 4 B bytes into one 32-bit reg each
    // (little-endian: byte v at bits [8v .. 8v+7]).
    unsigned int ar = 0, br = 0;
    int c0, c1, d0, d1;
#pragma unroll
    for (int v = 0; v < 4; ++v) {
        ar |= (static_cast<unsigned int>(static_cast<unsigned char>(A[g * 16 + tg * 4 + v]))) << (8 * v);
        br |= (static_cast<unsigned int>(static_cast<unsigned char>(B[g * 16 + tg * 4 + v]))) << (8 * v);
    }
    c0 = C[g * 8 + (tg * 2 + 0)];   // C[m=g][n=tg*2+0]
    c1 = C[g * 8 + (tg * 2 + 1)];   // C[m=g][n=tg*2+1]

    asm volatile(
        "mma.sync.aligned.m8n8k16.row.col.satfinite.s32.s8.s8.s32 "
        "{%0, %1}, {%2}, {%3}, {%4, %5};\n"
        : "=r"(d0), "=r"(d1)
        : "r"(ar), "r"(br), "r"(c0), "r"(c1));

    D[g * 8 + (tg * 2 + 0)] = d0;   // D[m=g][n=tg*2+0]
    D[g * 8 + (tg * 2 + 1)] = d1;   // D[m=g][n=tg*2+1]
}
