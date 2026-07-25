// IMMA.SP.16864 int8 2:4-structured-sparse tensor-core MMA fixture (one warp,
// single m16n8k64 sparse atom).
//
// CUTLASS 15_ampere_sparse_tensorop_gemm emits IMMA.SP.16864.S8.S8 (the sparse
// 2:4 int8 tensorop atom). The PTX is
//   mma.sp.sync.aligned.m16n8k64.row.col.s32.s8.s8.s32.satfinite
//      {D0..D3}, {A0..A3}, {B0..B3}, {C0..C3}, E, 0x0
// where A is SPARSE-COMPRESSED (32 of every 64 K-columns kept per row) and E is
// the metadata: for each group of 4 K-columns, a 4-bit field (idx0 | idx1<<2)
// selecting which 2 of the 4 dense positions the 2 kept values occupy.
//
// Each lane loads its fragments from plain host buffers using the CUTLASS cute
// SM80_16x8x32_S32S8S8S32 layout (column-major / M-major codomain — the SAME
// convention the dense IMMA.8816 & DMMA fixes use), then performs ONE mma.sp and
// scatters its 4 D fragments back. lane = t (0..31), t0 = t%4, t1 = t//4.
//
// Per-lane fragment maps (verified for every lane):
//   C/D (M16 x N8, int32, 4 regs): m=t1+8*(v/2), n=t0*2+(v%2)        v in 0..3
//   B   (N8 x K64, s8, 4 regs=16 s8): two stacked dense-k32 B atoms; for packed
//        value index p in 0..15: (n,k)=blayout_k32(t, p%8); k += 32*(p/8)
//   A-compressed (M16 x 32 kept-cols, s8, 4 regs=16 s8): packed value index p in
//        0..15 -> (m, ccol)=alayout_k32(t, p).
//   E (1 uint32): for each of this lane's 2 rows, 4 dense groups (the 8 ccols this
//        lane holds for that row form groups g where ccols {2g,2g+1} belong to g),
//        4 bits/group = (idx0 | idx1<<2). The HOST builds E in exactly this layout.
//
// The earlier x86 lifter stub copied C->D (the A*B product dropped). This fixture
// isolates the sparse atom so the warp-true wrapper imma_sp16864_s8_warp can be
// validated bit-exact against a host oracle that uncompresses A via the same E
// (cutlass host_uncompress.h rule) and computes D = C + A_dense * B^T.
//
// Param offsets (sm90, base 0x210): 0x210 A(comp), 0x218 B, 0x220 C, 0x228 D,
//                                   0x230 E (metadata).

#include <cstdint>

extern "C" __global__ void kern_imma_sp16864(
        const signed char* __restrict__ Acomp,   // M16 x 32 compressed, row-major s8
        const signed char* __restrict__ B,       // N8  x 64 dense,      row-major s8 (N x K)
        const int* __restrict__ C,                // M16 x N8 int32, row-major
        int* __restrict__ D,                      // M16 x N8 int32, row-major
        const uint32_t* __restrict__ Emeta)       // 32 lanes x 1 uint32 metadata
{
    const int lane = threadIdx.x & 31;
    const int t0 = lane & 3;       // lane%4
    const int t1 = lane >> 2;      // lane/4

    // --- gather A-compressed: 16 s8 -> 4 packed regs (little-endian bytes) ---
    // alayout_k32(t,p): off = 64*t0 + t1 + 16*(p%4) + 8*((p/4)%2) + 256*((p/8)%2)
    //                   m = off%16 ; ccol = off/16
    unsigned int ar[4] = {0, 0, 0, 0};
    unsigned int br[4] = {0, 0, 0, 0};
#pragma unroll
    for (int p = 0; p < 16; ++p) {
        const int off  = 64 * t0 + t1 + 16 * (p % 4) + 8 * ((p / 4) % 2) + 256 * ((p / 8) % 2);
        const int m    = off % 16;
        const int ccol = off / 16;
        ar[p >> 2] |= (static_cast<unsigned int>(
                          static_cast<unsigned char>(Acomp[m * 32 + ccol]))) << (8 * (p & 3));
        // B: two stacked dense-k32 atoms. blayout_k32(t, p%8):
        //   offB = 32*t0 + t1 + 8*((p%8)%4) + 128*((p%8)/4); n=offB%8; k=offB/8; k += 32*(p/8)
        const int ps   = p % 8;
        const int offB = 32 * t0 + t1 + 8 * (ps % 4) + 128 * (ps / 4);
        const int bn   = offB % 8;
        const int bk   = (offB / 8) + 32 * (p / 8);
        br[p >> 2] |= (static_cast<unsigned int>(
                          static_cast<unsigned char>(B[bn * 64 + bk]))) << (8 * (p & 3));
    }

    // --- gather C: 4 int32 regs ---  m=t1+8*(v/2), n=t0*2+(v%2)
    int cr[4], dr[4];
#pragma unroll
    for (int v = 0; v < 4; ++v) {
        const int m = t1 + 8 * (v / 2);
        const int n = t0 * 2 + (v % 2);
        cr[v] = C[m * 8 + n];
    }

    const uint32_t e = Emeta[lane];

    asm volatile(
        "mma.sp.sync.aligned.m16n8k64.row.col.s32.s8.s8.s32.satfinite "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, %16, 0x0;\n"
        : "=r"(dr[0]), "=r"(dr[1]), "=r"(dr[2]), "=r"(dr[3])
        : "r"(ar[0]), "r"(ar[1]), "r"(ar[2]), "r"(ar[3]),
          "r"(br[0]), "r"(br[1]), "r"(br[2]), "r"(br[3]),
          "r"(cr[0]), "r"(cr[1]), "r"(cr[2]), "r"(cr[3]),
          "r"(e));

#pragma unroll
    for (int v = 0; v < 4; ++v) {
        const int m = t1 + 8 * (v / 2);
        const int n = t0 * 2 + (v % 2);
        D[m * 8 + n] = dr[v];
    }
}
