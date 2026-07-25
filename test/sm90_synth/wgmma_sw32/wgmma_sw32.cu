// wgmma_sw32 — isolate FA3 bwd hdim96's dQ gemm: m64n48k16 SS wgmma with
// trans-b over an SW32-swizzled, TMA-loaded B tile.
//
// Replicates the captured configuration from the real kernel:
//   B: K-like [128k x 96n] fp16, six TMA boxes (16 cols x 128 rows, SW32)
//      landing at smem base + boxIdx*0x1000; desc lbo=0x1000 sbo=0x100 sw=3;
//      per-WG window: WG0 base+0x0 (cols 0-47), WG1 base+0x3000 (cols 48-95)
//      — here ONE warpgroup runs both windows sequentially.
//   A: dS-like [64m x 128k] fp16, plain K-major smem (no swizzle, KINTER
//      LBO16=64 SBO16=8 like wgmma_min) — the B side is the suspect; A is
//      kept simple so the CPU check pins B behavior.
//   8 k-slice wgmma m64n48k16 f32 chain, imm-trans-b=1, A desc +0x20/slice
//      (K-major 16-elt slices), B desc +0x200/slice.
//
// PASS criterion: kernel out == CPU reference for BOTH windows. The lifted
// PTX must match vanilla bit-for-bit (deterministic integer-free fp16 data).
#include <cstdio>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>

#define CK(x) do { cudaError_t e=(x); if(e){printf("CUDA err %s @%d\n",cudaGetErrorString(e),__LINE__); return 1;} } while(0)

__device__ __forceinline__ unsigned smem_u32(const void* p) {
  return (unsigned)__cvta_generic_to_shared(p);
}

__device__ __forceinline__ unsigned long long make_desc(unsigned base16b,
                                                        unsigned lbo,
                                                        unsigned sbo,
                                                        unsigned sw) {
  return ((unsigned long long)(base16b >> 4) & 0x3FFFull)
       | (((unsigned long long)(lbo >> 4) & 0x3FFFull) << 16)
       | (((unsigned long long)(sbo >> 4) & 0x3FFFull) << 32)
       | ((unsigned long long)(sw & 3) << 62);
}

// B tile: 24KB SW32 TMA-loaded; A tile: 16KB plain. One mbarrier.
extern __shared__ __align__(1024) unsigned char smem_raw[];

__global__ void kern(const __grid_constant__ CUtensorMap tmapB,
                     const __grid_constant__ CUtensorMap tmapA,
                     float* __restrict__ out /* [2][64][48] */) {
  __half* sB = (__half*)(smem_raw);            // 24KB SW32 K-tile
  __half* sA = (__half*)(smem_raw + 24*1024);  // 16KB A, SW128 TMA-loaded
  unsigned long long* mbar = (unsigned long long*)(smem_raw + 40*1024);
  const int tid = threadIdx.x;

  // ---- TMA-load B (six 16x128 SW32 boxes at +0x1000) and A (two 64x64
  // SW128 boxes at +0x2000) with one expected-tx barrier.
  if (tid == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(smem_u32(mbar)));
    asm volatile("fence.proxy.async.shared::cta;");
    unsigned mb = smem_u32(mbar);
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                 :: "r"(mb), "r"(6*4096 + 2*8192));
    for (int b = 0; b < 6; ++b) {
      unsigned dst = smem_u32(sB) + b*0x1000;
      int c0 = 16*b, c1 = 0;
      asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(dst), "l"(&tmapB), "r"(c0), "r"(c1), "r"(mb) : "memory");
    }
    for (int b = 0; b < 2; ++b) {
      unsigned dst = smem_u32(sA) + b*0x2000;
      int c0 = 64*b, c1 = 0;
      asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(dst), "l"(&tmapA), "r"(c0), "r"(c1), "r"(mb) : "memory");
    }
  }
  __syncthreads();
  {
    unsigned mb = smem_u32(mbar);
    asm volatile("{.reg .pred p; WAIT: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], 0; @!p bra WAIT;}"
                 :: "r"(mb));
  }
  __syncthreads();

  // ---- the gemm chain, one window per CONCURRENT warpgroup
  {
    const int wg = tid >> 7;
    float acc[24];
    unsigned aBase = smem_u32(sA);
    unsigned bBase = smem_u32(sB) + wg*0x3000;
    asm volatile("wgmma.fence.sync.aligned;");
    for (int s = 0; s < 8; ++s) {
      // A: SW128, matches the captured FA3 desc: lbo 0x10, sbo 0x400, sw1,
      // +0x20 per k16 slice with a +0x2000 block jump every 4 slices.
      unsigned long long dA = make_desc(aBase + (s/4)*0x2000 + (s%4)*0x20, 0x10, 0x400, 1);
      unsigned long long dB = make_desc(bBase + s*0x200, 0x1000, 0x100, 3);
      if (s == 0)
        asm volatile(
          "{wgmma.mma_async.sync.aligned.m64n48k16.f32.f16.f16 "
          "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23},"
          " %24, %25, 0, 1, 1, 0, 1;}"
          : "=f"(acc[0]),"=f"(acc[1]),"=f"(acc[2]),"=f"(acc[3]),"=f"(acc[4]),"=f"(acc[5]),
            "=f"(acc[6]),"=f"(acc[7]),"=f"(acc[8]),"=f"(acc[9]),"=f"(acc[10]),"=f"(acc[11]),
            "=f"(acc[12]),"=f"(acc[13]),"=f"(acc[14]),"=f"(acc[15]),"=f"(acc[16]),"=f"(acc[17]),
            "=f"(acc[18]),"=f"(acc[19]),"=f"(acc[20]),"=f"(acc[21]),"=f"(acc[22]),"=f"(acc[23])
          : "l"(dA), "l"(dB));
      else
        asm volatile(
          "{wgmma.mma_async.sync.aligned.m64n48k16.f32.f16.f16 "
          "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23},"
          " %24, %25, 1, 1, 1, 0, 1;}"
          : "+f"(acc[0]),"+f"(acc[1]),"+f"(acc[2]),"+f"(acc[3]),"+f"(acc[4]),"+f"(acc[5]),
            "+f"(acc[6]),"+f"(acc[7]),"+f"(acc[8]),"+f"(acc[9]),"+f"(acc[10]),"+f"(acc[11]),
            "+f"(acc[12]),"+f"(acc[13]),"+f"(acc[14]),"+f"(acc[15]),"+f"(acc[16]),"+f"(acc[17]),
            "+f"(acc[18]),"+f"(acc[19]),"+f"(acc[20]),"+f"(acc[21]),"+f"(acc[22]),"+f"(acc[23])
          : "l"(dA), "l"(dB));
    }
    asm volatile("wgmma.commit_group.sync.aligned;");
    asm volatile("wgmma.wait_group.sync.aligned 0;");

    // canonical m64nN f32 accumulator writeback (validated in wgmma_min):
    // lane l of warp w: rows 16w + l/4 and +8; reg pair 2i,2i+1 -> cols
    // 8i + (l%4)*2 (+1)
    int w = (tid >> 5) & 3, l = tid & 31;
    for (int i = 0; i < 12; ++i) {
      int row = 16*w + (l >> 2) + (i & 1)*8;
      int col = 8*(i >> 1) + (l & 3)*2;
      out[(wg*64 + row)*48 + col]     = acc[2*i];
      out[(wg*64 + row)*48 + col + 1] = acc[2*i+1];
    }
    asm volatile("wgmma.fence.sync.aligned;");
  }
}

int main() {
  // deterministic data
  std::vector<__half> hA(64*128), hB(128*96);
  for (int m = 0; m < 64; ++m)
    for (int k = 0; k < 128; ++k)
      hA[m*128+k] = __float2half(((m*7+k*3) % 23 - 11) * 0.0625f);
  for (int k = 0; k < 128; ++k)
    for (int n = 0; n < 96; ++n)
      hB[k*96+n] = __float2half(((k*5+n*11) % 19 - 9) * 0.0625f);

  __half *dA, *dB; float* dOut;
  CK(cudaMalloc(&dA, hA.size()*2)); CK(cudaMalloc(&dB, hB.size()*2));
  CK(cudaMalloc(&dOut, 2*64*48*4)); CK(cudaMemset(dOut, 0, 2*64*48*4));
  CK(cudaMemcpy(dA, hA.data(), hA.size()*2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dB, hB.data(), hB.size()*2, cudaMemcpyHostToDevice));

  // tensormap: global B [128 rows x 96 cols] fp16 row-major; boxes 128x16,
  // inner (contiguous) dim = cols. cuTensorMapEncodeTiled dims are
  // inner-first: globalDim {96, 128}, boxDim {16, 128}, SW32.
  CUtensorMap tmap, tmapA;
  cuuint64_t gdim[2] = {96, 128};
  cuuint64_t gstride[1] = {96*2};          // bytes between rows
  cuuint32_t box[2] = {16, 128};
  cuuint32_t estride[2] = {1, 1};
  CUresult r = cuTensorMapEncodeTiled(
      &tmap, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2, dB, gdim, gstride, box,
      estride, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_32B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS) { printf("tmapB encode failed %d\n", r); return 1; }
  cuuint64_t gdimA[2] = {128, 64};
  cuuint64_t gstrideA[1] = {128*2};
  cuuint32_t boxA[2] = {64, 64};
  r = cuTensorMapEncodeTiled(
      &tmapA, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2, dA, gdimA, gstrideA, boxA,
      estride, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS) { printf("tmapA encode failed %d\n", r); return 1; }

  CK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, 48*1024));
  kern<<<1, 256, 48*1024>>>(tmap, tmapA, dOut);
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());

  std::vector<float> got(2*64*48);
  CK(cudaMemcpy(got.data(), dOut, got.size()*4, cudaMemcpyDeviceToHost));

  // CPU reference: D[wg][m][j] = sum_k A[m,k] * B[k, wg*48 + j]
  int bad = 0;
  for (int wg = 0; wg < 2; ++wg)
    for (int m = 0; m < 64; ++m)
      for (int j = 0; j < 48; ++j) {
        float ref = 0;
        for (int k = 0; k < 128; ++k)
          ref += __half2float(hA[m*128+k]) * __half2float(hB[k*96 + wg*48 + j]);
        float g = got[(wg*64+m)*48+j];
        if (fabsf(ref - g) > 0.01f * (1 + fabsf(ref))) {
          if (bad < 8) printf("MISMATCH wg%d m%d j%d: got %f want %f\n", wg, m, j, g, ref);
          ++bad;
        }
      }
  printf(bad ? "FAILED (%d bad)\n" : "TEST PASSED\n", bad);
  return bad != 0;
}
