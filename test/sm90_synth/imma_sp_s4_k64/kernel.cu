// IMMA.SP.16864 int4 2:4-structured-sparse tensor-core MMA fixture (one warp,
// single LOGICAL m16n8k128 int4 sparse atom).
//
// CUTLASS 15_ampere_sparse_tensorop_gemm is an INT4 (integer_subbyte<4>) 2:4-sparse
// GEMM whose warp atom is SparseMma<GemmShape<16,8,128>,32,int4b_t,...>. Its PTX is
//   mma.sp.sync.aligned.m16n8k128.row.col.s32.s4.s4.s32.satfinite
//      {D0..D3}, {A0..A3}, {B0..B3}, {C0..C3}, E, 0x0
// nvcc lowers this ONE PTX op to a PAIR of IMMA.SP.16864.S8.S8 SASS sub-atoms (the int4
// m16n8k128 atom is emitted under the .S8.S8 mnemonic family — SASS text is identical to
// the int8 m16n8k64 atom; the int4 dtype/shape live only in the kernel-symbol metadata).
//
// Fragments (cute SM80_16x8x64_S32S4S4S32_TN strides reused for the sparse operands,
// column/M-major codomain — the SAME convention the int8 IMMA.SP fix used):
//   FragmentA = Array<int4,32> = 4 b32 regs (8 int4/reg). A SPARSE-COMPRESSED: 32 kept
//      int4 of K=128 per row.
//   FragmentB = Array<int4,32> = 4 b32 regs. B DENSE (N8 x K128 = two stacked dense-k64).
//   FragmentC = FragmentD = Array<int,4> = 4 s32 regs.
//   FragmentE = uint32 (1 reg) = metadata; for int4: each 4-bit nibble governs a group of
//      8 dense int4 (= 4 PAIRS), idx0|idx1<<2 selecting which 2 of the 4 pairs are kept
//      (cutlass host_uncompress.h int4 rule: step=8, a=4, b=8, ElementsPerE=2).
//
// Per-lane fragment maps (verified for every lane).
//   lane = t (0..31), t0 = t%4, t1 = t//4.
//   C/D (M16 x N8, int32, 4 regs): m=t1+8*(v/2), n=t0*2+(v%2)              v in 0..3
//   A-comp (M16 x 64 kept, int4, 4 regs=32 int4): packed nibble index p in 0..31:
//        off = 128*t0 + t1 + 16*(p%8) + 8*((p/8)%2) + 512*((p/16)%2); m=off%16; ccol=off/16
//   B (N8 x K128 dense, int4, 4 regs=32 int4): two stacked dense-k64 atoms; for packed
//        nibble index p in 0..31: ps=p%16; offB=64*t0+t1+8*(ps%8)+256*(ps/8); n=offB%8;
//        k=offB/8 + 64*(p/16)
//
// E layout (host + kernel + wrapper all agree): a lane's 32 A-comp int4 form 16 PAIRS
//   (pair j = nibbles {2j,2j+1}); each pair is the kept pair of a group g (g = ccol/4 of
//   that pair) for row m. The lane's uint32 E packs, for each of its 8 nibble-GROUPS (each
//   group = 4 dense pairs = one host_uncompress nibble), the (idx0|idx1<<2) selection.
//   Because the lane holds 4 regs x (8 int4) and a group-of-8-dense maps to 4 compressed
//   int4 (2 pairs) = ONE reg's worth split, the lane's 8 E-nibbles correspond to its 8
//   (ccol//4) groups across its 2 rows. The HOST builds E in exactly this layout.
//
// On x86 the HW mma.sp NEVER executes — the lifter replaces it with imma_sp_s4_warp, which
// reconstructs dense int4 A/B from these same per-lane fragments + E and computes
// D = C + A_dense * B^T (sat s32). The host oracle uncompresses A from the host buffers via
// the SAME int4 rule and computes the same GEMM, so wrapper==oracle is bit-exact.
//
// Param offsets (sm90, base 0x210): 0x210 A(comp), 0x218 B, 0x220 C, 0x228 D, 0x230 E.

#include <cstdint>

// NOTE on the kernel name: the int4-vs-int8 disambiguation keys on the kernel SYMBOL
// (the only place the int4 dtype/shape survive — SASS text is identical to int8). The real
// CUTLASS 15_sparse symbol carries `integer_subbyteILi4ELb1EE` (int4 elem) +
// `ILi16ELi8ELi128EE` (the SparseMma<GemmShape<16,8,128>> instruction shape). This microbench
// embeds BOTH substrings in its (extern "C") name so it exercises the SAME lifter detection
// path as the real kernel — no microbench-only special-case in the lifter.
extern "C" __global__ void
kern_imma_sp_s4_integer_subbyteILi4ELb1EE_SparseMmaINS6_ILi16ELi8ELi128EE(
        const uint32_t* __restrict__ Acomp,  // 16 rows x 64 kept int4, packed 8/reg, row-major:
                                             //   Acomp[m*8 + (ccol/8)] holds ccol%8 nibble
        const uint32_t* __restrict__ B,      // 8 rows x 128 int4, packed 8/reg, row-major (N x K):
                                             //   B[n*16 + (k/8)] holds k%8 nibble
        const int* __restrict__ C,           // M16 x N8 int32, row-major
        int* __restrict__ D,                 // M16 x N8 int32, row-major
        const uint32_t* __restrict__ Emeta)  // 32 lanes x 1 uint32 metadata
{
    const int lane = threadIdx.x & 31;
    const int t0 = lane & 3;
    const int t1 = lane >> 2;

    auto getA = [&](int m, int ccol) -> uint32_t {       // ccol in 0..63 -> nibble
        uint32_t w = Acomp[m * 8 + (ccol >> 3)];
        return (w >> (4 * (ccol & 7))) & 0xf;
    };
    auto getB = [&](int n, int k) -> uint32_t {          // k in 0..127 -> nibble
        uint32_t w = B[n * 16 + (k >> 3)];
        return (w >> (4 * (k & 7))) & 0xf;
    };

    // --- gather A-compressed: 32 int4 -> 4 packed regs ---
    uint32_t ar[4] = {0, 0, 0, 0};
    uint32_t br[4] = {0, 0, 0, 0};
#pragma unroll
    for (int p = 0; p < 32; ++p) {
        const int offA  = 128 * t0 + t1 + 16 * (p % 8) + 8 * ((p / 8) % 2) + 512 * ((p / 16) % 2);
        const int m     = offA % 16;
        const int ccol  = offA / 16;
        ar[p >> 3] |= (getA(m, ccol) & 0xf) << (4 * (p & 7));

        const int ps    = p % 16;
        const int offB  = 64 * t0 + t1 + 8 * (ps % 8) + 256 * (ps / 8);
        const int bn    = offB % 8;
        const int bk    = (offB / 8) + 64 * (p / 16);
        br[p >> 3] |= (getB(bn, bk) & 0xf) << (4 * (p & 7));
    }

    int cr[4], dr[4];
#pragma unroll
    for (int v = 0; v < 4; ++v) {
        const int m = t1 + 8 * (v / 2);
        const int n = t0 * 2 + (v % 2);
        cr[v] = C[m * 8 + n];
    }

    const uint32_t e = Emeta[lane];

    asm volatile(
        "mma.sp.sync.aligned.m16n8k128.row.col.s32.s4.s4.s32.satfinite "
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
