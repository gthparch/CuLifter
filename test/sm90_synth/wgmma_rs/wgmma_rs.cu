// wgmma_rs — minimal Hopper RS-form warpgroup-MMA microbench (FA3 GEMM1 shape).
// One warpgroup computes D2[64x64] = A[64x16] * B[16x64] where A comes from
// REGISTERS (the FA3 P@V pattern) and B from shared memory via a GMMA
// descriptor:  SASS HGMMA.64x64x16.F32 Rd, Ra, gdesc[URn], Rc.
//
// The A-fragment registers are produced the way FlashAttention-3 produces P:
// a preliminary SS wgmma against an identity B (D = A @ I -> A values in the
// ACCUMULATOR fragment layout), then f32->f16 pairwise packing — the same
// convert_layout_acc_Aregs identity cute relies on (accumulator layout of
// n16 == A-fragment layout of k16). No hand-derived fragment indexing.
//
// Deterministic inputs; the host verifies against a CPU reference matmul.
// vanilla-vs-lifted must both print the same PASS checksum.
#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__device__ __forceinline__ unsigned smem_u32(const void* p) {
  return (unsigned)__cvta_generic_to_shared(p);
}

// GMMA shared-memory matrix descriptor, no-swizzle K-INTER layout
// (validated in wgmma_min: LBO16=64, SBO16=8).
__device__ __forceinline__ unsigned long long make_desc(unsigned base,
                                                        unsigned lbo,
                                                        unsigned sbo) {
  return ((unsigned long long)(base >> 4) & 0x3FFFull)
       | (((unsigned long long)lbo & 0x3FFFull) << 16)
       | (((unsigned long long)sbo & 0x3FFFull) << 32);
}

__global__ void wgmma_rs_kernel(const __half* __restrict__ A,
                                const __half* __restrict__ B,
                                float* __restrict__ C) {
  __shared__ alignas(128) __half sA[64 * 16];   // A 64x16 K-major
  __shared__ alignas(128) __half sI[64 * 16];   // identity B0 (n=64, k=16)
  __shared__ alignas(128) __half sB[64 * 16];   // B 64(n) x 16(k) n-major
  const int tid = threadIdx.x;

  for (int idx = tid; idx < 64 * 16; idx += blockDim.x) {
    int i = idx / 16, k = idx % 16;
    int off = (i / 8) * 64 + (i % 8) * 8 + (k % 8) + (k / 8) * 512;
    sA[off] = A[i * 16 + k];
    sB[off] = B[i * 16 + k];
    // B0[n,k] = (n%16 == k) * (1 + n/16): acc cols 0-15 = A, cols 16-31 = 2A,
    // ... — each 16-col group is a scaled copy of A, so the two packed
    // A-fragments (quads) differ by a factor and an off-by-one quad binding
    // in a CHAINED RS gemm is numerically detectable.
    sI[off] = __float2half((i % 16 == k) ? (1.0f + i / 16) : 0.0f);
  }
  __syncthreads();
  asm volatile("fence.proxy.async.shared::cta;");

  unsigned long long dA = make_desc(smem_u32(sA), 64, 8);
  unsigned long long dI = make_desc(smem_u32(sI), 64, 8);
  unsigned long long dB = make_desc(smem_u32(sB), 64, 8);

  // GEMM0 (SS): acc[n] = A @ I  -> acc columns n<16 hold A[m, n].
  float acc[32];
  asm volatile("wgmma.fence.sync.aligned;");
  asm volatile(
      "{wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, "
      "%32, %33, 0, 1, 1, 0, 0;}"
      : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3]),
        "=f"(acc[4]), "=f"(acc[5]), "=f"(acc[6]), "=f"(acc[7]),
        "=f"(acc[8]), "=f"(acc[9]), "=f"(acc[10]), "=f"(acc[11]),
        "=f"(acc[12]), "=f"(acc[13]), "=f"(acc[14]), "=f"(acc[15]),
        "=f"(acc[16]), "=f"(acc[17]), "=f"(acc[18]), "=f"(acc[19]),
        "=f"(acc[20]), "=f"(acc[21]), "=f"(acc[22]), "=f"(acc[23]),
        "=f"(acc[24]), "=f"(acc[25]), "=f"(acc[26]), "=f"(acc[27]),
        "=f"(acc[28]), "=f"(acc[29]), "=f"(acc[30]), "=f"(acc[31])
      : "l"(dA), "l"(dI));
  asm volatile("wgmma.commit_group.sync.aligned;");
  asm volatile("wgmma.wait_group.sync.aligned 0;");

  // Pack the first 32 accumulator columns (acc[0..15]) into TWO k16
  // A-fragments: 8 x .b32, each = {hi: f16(acc[2j+1]), lo: f16(acc[2j])}.
  // IN-PLACE form: dest register == low source (the reg still holds the f32
  // accumulator; cvt reads it, converts, overwrites) — this is exactly FA3's
  // `F2FP.F16.F32.PACK_AB R216, R11, R216` spelling, which the out-of-place
  // cvt never exercises (in-place def+use of the SAME register through
  // SSA/def-use is the suspect path).
  unsigned areg[8];
  for (int j = 0; j < 8; ++j) {
    areg[j] = __float_as_uint(acc[2 * j]);        // reg holds f32 (low src)
    asm("cvt.rn.f16x2.f32 %0, %1, %0;"
        : "+r"(areg[j]) : "f"(acc[2 * j + 1]));   // in-place PACK_AB
  }

  // GEMM1 (RS): D2 = A0(64x16) @ B(16x64), then chained accumulate
  // GEMM2 (RS): D2 += A1 @ B with A1 = 2*A0 (from the scaled identity), so
  // D2 must equal 3*A@B. A one-quad-behind A binding (the FA3 symptom)
  // yields 2*A@B instead — numerically detectable. GEMM2 also exercises the
  // tied-C accumulate emission (scale-d=1).
  float d2[32];
  asm volatile("wgmma.fence.sync.aligned;");
  asm volatile(
      "{wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, "
      "{%32,%33,%34,%35}, %36, 0, 1, 1, 0;}"
      : "=f"(d2[0]), "=f"(d2[1]), "=f"(d2[2]), "=f"(d2[3]),
        "=f"(d2[4]), "=f"(d2[5]), "=f"(d2[6]), "=f"(d2[7]),
        "=f"(d2[8]), "=f"(d2[9]), "=f"(d2[10]), "=f"(d2[11]),
        "=f"(d2[12]), "=f"(d2[13]), "=f"(d2[14]), "=f"(d2[15]),
        "=f"(d2[16]), "=f"(d2[17]), "=f"(d2[18]), "=f"(d2[19]),
        "=f"(d2[20]), "=f"(d2[21]), "=f"(d2[22]), "=f"(d2[23]),
        "=f"(d2[24]), "=f"(d2[25]), "=f"(d2[26]), "=f"(d2[27]),
        "=f"(d2[28]), "=f"(d2[29]), "=f"(d2[30]), "=f"(d2[31])
      : "r"(areg[0]), "r"(areg[1]), "r"(areg[2]), "r"(areg[3]), "l"(dB));
  asm volatile(
      "{wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, "
      "{%32,%33,%34,%35}, %36, 1, 1, 1, 0;}"
      : "+f"(d2[0]), "+f"(d2[1]), "+f"(d2[2]), "+f"(d2[3]),
        "+f"(d2[4]), "+f"(d2[5]), "+f"(d2[6]), "+f"(d2[7]),
        "+f"(d2[8]), "+f"(d2[9]), "+f"(d2[10]), "+f"(d2[11]),
        "+f"(d2[12]), "+f"(d2[13]), "+f"(d2[14]), "+f"(d2[15]),
        "+f"(d2[16]), "+f"(d2[17]), "+f"(d2[18]), "+f"(d2[19]),
        "+f"(d2[20]), "+f"(d2[21]), "+f"(d2[22]), "+f"(d2[23]),
        "+f"(d2[24]), "+f"(d2[25]), "+f"(d2[26]), "+f"(d2[27]),
        "+f"(d2[28]), "+f"(d2[29]), "+f"(d2[30]), "+f"(d2[31])
      : "r"(areg[4]), "r"(areg[5]), "r"(areg[6]), "r"(areg[7]), "l"(dB));
  asm volatile("wgmma.commit_group.sync.aligned;");
  asm volatile("wgmma.wait_group.sync.aligned 0;");

  // Canonical wgmma m64n64 f32 accumulator (row,col) mapping (wgmma_min).
  for (int i = 0; i < 32; ++i) {
    int r = 16 * (tid / 32) + 8 * ((i % 4) / 2) + (tid % 32) / 4;
    int c = 8 * (i / 4) + (tid % 4) * 2 + (i % 2);
    C[r * 64 + c] = d2[i];
  }
}

int main() {
  const int MK = 64 * 16, NK = 64 * 16, MN = 64 * 64;
  __half *A_h = new __half[MK], *B_h = new __half[NK];
  for (int i = 0; i < MK; ++i) A_h[i] = __float2half((float)((i * 13 + 7) % 17) * 0.125f);
  for (int i = 0; i < NK; ++i) B_h[i] = __float2half((float)((i * 5 + 3) % 19) * 0.0625f);
  __half *A_d, *B_d; float* C_d;
  cudaMalloc(&A_d, MK * 2); cudaMalloc(&B_d, NK * 2); cudaMalloc(&C_d, MN * 4);
  cudaMemcpy(A_d, A_h, MK * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, B_h, NK * 2, cudaMemcpyHostToDevice);
  wgmma_rs_kernel<<<1, 128>>>(A_d, B_d, C_d);
  if (cudaDeviceSynchronize() != cudaSuccess) { printf("CUDA ERROR\n"); return 1; }
  float* C_hd = new float[MN];
  cudaMemcpy(C_hd, C_d, MN * 4, cudaMemcpyDeviceToHost);

  // CPU reference: D2[m,n] = 3 * sum_k A[m,k] * B[n,k]  (A0@B + 2A0@B;
  // B stored n-major 64x16).
  int bad = 0;
  double checksum = 0;
  for (int m = 0; m < 64; ++m)
    for (int n = 0; n < 64; ++n) {
      float ref = 0;
      for (int k = 0; k < 16; ++k)
        ref += 3.0f * __half2float(__half(A_h[m * 16 + k])) * __half2float(__half(B_h[n * 16 + k]));
      float got = C_hd[m * 64 + n];
      checksum += got * ((m * 64 + n) % 97 + 1);
      if (fabsf(ref - got) > 0.02f * (1 + fabsf(ref))) {
        if (bad < 5) printf("MISMATCH m=%d n=%d ref=%.4f got=%.4f\n", m, n, ref, got);
        ++bad;
      }
    }
  printf("checksum %.4f\n", checksum);
  printf(bad ? "TEST FAILED (%d bad)\n" : "TEST PASSED\n", bad);
  return bad ? 1 : 0;
}
