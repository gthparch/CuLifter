// derive_desc — one-off: find the GMMA no-swizzle descriptor (LBO,SBO) and
// smem element layout for m64n64k16 f32.bf16.bf16 SS wgmma by comparing
// against a CPU reference. Prints every (layoutA,layoutB,LBO,SBO) combo whose
// max-abs-err < 1e-3.
#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <algorithm>

__device__ __forceinline__ unsigned smem_u32(const void* p) {
  return (unsigned)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ unsigned long long make_desc(unsigned base,
                                                        unsigned lbo16,
                                                        unsigned sbo16) {
  return ((unsigned long long)((base >> 4) & 0x3FFF))
       | (((unsigned long long)(lbo16 & 0x3FFF)) << 16)
       | (((unsigned long long)(sbo16 & 0x3FFF)) << 32);
}

// layout modes for element (i,k) of a 64x16 tile, i=mn index, k=k index:
// 0: linear K-major            i*16 + k
// 1: core-matrix K-major       (i%8)*8 + (k%8) + (i/8)*64*... -> canonical:
//    core (ci,cj), ci=i/8, cj=k/8, in-core (i%8)*8+(k%8); cores laid ci-major:
//    ((i/8)*2 + (k/8))*64 + (i%8)*8 + (k%8)
// 2: cores k-major first: ((k/8)*8 + (i/8))*64 + (i%8)*8 + (k%8)
__device__ int elem_off(int mode, int i, int k) {
  if (mode == 0) return i * 16 + k;
  if (mode == 1) return ((i / 8) * 2 + (k / 8)) * 64 + (i % 8) * 8 + (k % 8);
  if (mode == 2) return ((k / 8) * 8 + (i / 8)) * 64 + (i % 8) * 8 + (k % 8);
  // 3: canonical no-swizzle: 8x8 cores contiguous, k-cores inner, m-cores outer
  if (mode == 3) return (i / 8) * 128 + (k / 8) * 64 + (i % 8) * 8 + (k % 8);
  // 4: cute Layout_K_INTER canonical: atom 8x8 row-major, tiled m-first then k
  //    offset = (i/8)*64 + (i%8)*8 + (k%8) + (k/8)*512  [elements]
  return (i / 8) * 64 + (i % 8) * 8 + (k % 8) + (k / 8) * 512;
}

__global__ void try_combo(const __nv_bfloat16* A, const __nv_bfloat16* B,
                          float* C, int modeA, int modeB, int lbo16, int sbo16) {
  __shared__ alignas(1024) __nv_bfloat16 sA[64 * 16];
  __shared__ alignas(1024) __nv_bfloat16 sB[64 * 16];
  const int tid = threadIdx.x;
  for (int idx = tid; idx < 64 * 16; idx += blockDim.x) {
    int i = idx / 16, k = idx % 16;
    sA[elem_off(modeA, i, k)] = A[i * 16 + k];   // A row-major source
    sB[elem_off(modeB, i, k)] = B[i * 16 + k];   // B: i = n index
  }
  __syncthreads();
  asm volatile("fence.proxy.async.shared::cta;");  // generic->async proxy: make st.shared visible to GMMA
  unsigned long long dA = make_desc(smem_u32(sA), lbo16, sbo16);
  unsigned long long dB = make_desc(smem_u32(sB), lbo16, sbo16);
  float acc[32];
  asm volatile("wgmma.fence.sync.aligned;");
  asm volatile(
      "{wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
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
      : "l"(dA), "l"(dB));
  asm volatile("wgmma.commit_group.sync.aligned;");
  asm volatile("wgmma.wait_group.sync.aligned 0;");
  for (int i = 0; i < 32; ++i) C[tid * 32 + i] = acc[i];
}

int main() {
  const int MK = 64 * 16, MN = 64 * 64;
  __nv_bfloat16 *A_h = new __nv_bfloat16[MK], *B_h = new __nv_bfloat16[MK];
  float *Af = new float[MK], *Bf = new float[MK];
  for (int i = 0; i < MK; ++i) {
    Af[i] = (float)((i * 13 + 7) % 17) * 0.125f;
    Bf[i] = (float)((i * 5 + 3) % 19) * 0.0625f;
    A_h[i] = __float2bfloat16(Af[i]);
    B_h[i] = __float2bfloat16(Bf[i]);
  }
  // CPU ref: C[m][n] = sum_k A[m][k]*B[n][k]  (B n-major, same as A layout)
  float* ref = new float[MN];
  float* refs = new float[MN];
  for (int m = 0; m < 64; ++m)
    for (int n = 0; n < 64; ++n) {
      float s = 0;
      for (int k = 0; k < 16; ++k)
        s += __bfloat162float(A_h[m * 16 + k]) * __bfloat162float(B_h[n * 16 + k]);
      ref[m * 64 + n] = s;
    }
  for (int i = 0; i < MN; ++i) refs[i] = ref[i];
  std::sort(refs, refs + MN);
  __nv_bfloat16 *A_d, *B_d; float* C_d;
  cudaMalloc(&A_d, MK * 2); cudaMalloc(&B_d, MK * 2); cudaMalloc(&C_d, MN * 4);
  cudaMemcpy(A_d, A_h, MK * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, B_h, MK * 2, cudaMemcpyHostToDevice);
  float* C_h = new float[MN];
  int lbos[] = {1, 2, 4, 8, 16, 32, 64, 128, 256};
  int sbos[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512};
  for (int mA = 0; mA < 5; ++mA) for (int mB = 0; mB < 5; ++mB)
    for (int li = 0; li < 9; ++li) for (int si = 0; si < 10; ++si) {
      cudaMemset(C_d, 0, MN * 4);
      try_combo<<<1, 128>>>(A_d, B_d, C_d, mA, mB, lbos[li], sbos[si]);
      if (cudaDeviceSynchronize() != cudaSuccess) { cudaGetLastError(); continue; }
      cudaMemcpy(C_h, C_d, MN * 4, cudaMemcpyDeviceToHost);
      std::sort(C_h, C_h + MN);
      float maxerr = 0;
      for (int i = 0; i < MN; ++i) {
        float e = fabsf(C_h[i] - refs[i]);
        if (e > maxerr) maxerr = e;
      }
      if (maxerr < 1e-3f)
        printf("MATCH modeA=%d modeB=%d LBO16=%d SBO16=%d (maxerr %.5f)\n",
               mA, mB, lbos[li], sbos[si], maxerr);
      else if (maxerr < 0.5f)
        printf("close modeA=%d modeB=%d LBO16=%d SBO16=%d (maxerr %.5f)\n",
               mA, mB, lbos[li], sbos[si], maxerr);
    }
  printf("derive done\n");
  return 0;
}
