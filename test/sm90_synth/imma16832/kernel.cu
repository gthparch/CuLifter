// IMMA.16832 int8 tensor-core MMA fixture (one warp, single m16n8k32 atom).
//
// Each lane loads its A/B/C fragments from plain buffers using the CUTLASS cute
// MMA_Traits<SM80_16x8x32_S32S8S8S32_TN> mapping, performs ONE
// mma.sync.m16n8k32 (s32 = s8 * s8 + s32, satfinite), and scatters its 4 D
// fragments back.  This is the Ampere sm80 dense int8 atom (13_two_tensor_op_
// fusion s8 variants issue 412-544 of these) and the larger-shape sibling of the
// IMMA.8816 atom — both were stubbed on x86 as a C->D identity (A*B dropped).
//
// Fragment->matrix map (g = lane/4 in 0..7 = row group; tg = lane%4 in 0..3):
//   A: 4 regs, each 4 packed s8.  reg r, byte b in 0..3:
//     r0: A[m=g  ][k=     tg*4+b]   r1: A[m=g+8][k=     tg*4+b]
//     r2: A[m=g  ][k=16 + tg*4+b]   r3: A[m=g+8][k=16 + tg*4+b]   (A is M16 x K32 row-major)
//   B: 2 regs, each 4 packed s8.  reg r, byte b:
//     r0: B[n=g][k=     tg*4+b]     r1: B[n=g][k=16 + tg*4+b]     (B is N8 x K32, col-major K x N)
//   C/D: 4 s32 regs:
//     v0: C[m=g  ][n=tg*2+0]  v1: C[m=g  ][n=tg*2+1]
//     v2: C[m=g+8][n=tg*2+0]  v3: C[m=g+8][n=tg*2+1]              (C/D M16 x N8 int32)
//   int8 byte order: byte b at bits[8b..8b+7] (little-endian, matches __dp4a packing).
// Result: D[m][n] = C[m][n] + sum_{k<32} A[m][k]*B[n][k].
//
// (cute column-major (M-major) codomain — SAME convention as IMMA.8816 / DMMA.8x8x4.
//  A single atom already exercises the M16 split (rows g vs g+8) and the K32 split
//  (k<16 vs k>=16); a multi-tile variant additionally exposes any m<->n transpose.)
extern "C" __global__ void kern_imma16832(const signed char* A, const signed char* B,
                                          const int* C, int* D)
{
    const int lane = threadIdx.x & 31;
    const int g  = lane >> 2;   // lane/4 in {0..7}
    const int tg = lane & 3;    // lane%4 in {0..3}

    // Pack this lane's A (4 regs) and B (2 regs) int8 fragments.
    unsigned int a0 = 0, a1 = 0, a2 = 0, a3 = 0, b0 = 0, b1 = 0;
#pragma unroll
    for (int b = 0; b < 4; ++b) {
        a0 |= (static_cast<unsigned int>(static_cast<unsigned char>(A[(g    ) * 32 + (     tg * 4 + b)]))) << (8 * b);
        a1 |= (static_cast<unsigned int>(static_cast<unsigned char>(A[(g + 8) * 32 + (     tg * 4 + b)]))) << (8 * b);
        a2 |= (static_cast<unsigned int>(static_cast<unsigned char>(A[(g    ) * 32 + (16 + tg * 4 + b)]))) << (8 * b);
        a3 |= (static_cast<unsigned int>(static_cast<unsigned char>(A[(g + 8) * 32 + (16 + tg * 4 + b)]))) << (8 * b);
        b0 |= (static_cast<unsigned int>(static_cast<unsigned char>(B[(g    ) * 32 + (     tg * 4 + b)]))) << (8 * b);
        b1 |= (static_cast<unsigned int>(static_cast<unsigned char>(B[(g    ) * 32 + (16 + tg * 4 + b)]))) << (8 * b);
    }
    int c0 = C[(g    ) * 8 + (tg * 2 + 0)];
    int c1 = C[(g    ) * 8 + (tg * 2 + 1)];
    int c2 = C[(g + 8) * 8 + (tg * 2 + 0)];
    int c3 = C[(g + 8) * 8 + (tg * 2 + 1)];
    int d0, d1, d2, d3;

    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "r"(c0), "r"(c1), "r"(c2), "r"(c3));

    D[(g    ) * 8 + (tg * 2 + 0)] = d0;
    D[(g    ) * 8 + (tg * 2 + 1)] = d1;
    D[(g + 8) * 8 + (tg * 2 + 0)] = d2;
    D[(g + 8) * 8 + (tg * 2 + 1)] = d3;
}
