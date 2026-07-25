// probe_elem — element-level wgmma probe: A = identity (64x16 slab: I in the
// first 16 cols), so D = B^T rows land literally in the accumulators. Dumps
// thread 0/1/4 accumulators + smem-read expectations to pin the layout HW
// actually reads, one field at a time.
#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

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
// cute Layout_K_INTER canonical (mode 4)
__device__ int elem_off(int i, int k) {
  return (i / 8) * 64 + (i % 8) * 8 + (k % 8) + (k / 8) * 512;
}

__global__ void probe(float* out, int lbo16, int sbo16) {
  __shared__ alignas(1024) __nv_bfloat16 sA[64 * 16];
  __shared__ alignas(1024) __nv_bfloat16 sB[64 * 16];
  const int tid = threadIdx.x;
  for (int idx = tid; idx < 64 * 16; idx += blockDim.x) {
    int i = idx / 16, k = idx % 16;
    sA[elem_off(i, k)] = __float2bfloat16((i == k) ? 1.f : 0.f);  // A = I (top 16 rows)
    sB[elem_off(i, k)] = __float2bfloat16((float)(i * 100 + k));  // B[n][k] = n*100+k
  }
  __syncthreads();
  asm volatile("fence.proxy.async.shared::cta;");
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
  for (int i = 0; i < 32; ++i) out[tid * 32 + i] = acc[i];
}

int main(int argc, char** argv) {
  int lbo16 = argc > 1 ? atoi(argv[1]) : 64;
  int sbo16 = argc > 2 ? atoi(argv[2]) : 8;
  float* out_d; cudaMalloc(&out_d, 128 * 32 * 4);
  cudaMemset(out_d, 0, 128 * 32 * 4);
  probe<<<1, 128>>>(out_d, lbo16, sbo16);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) { printf("CUDA ERROR: %s\n", cudaGetErrorString(err)); return 1; }
  float* out_h = new float[128 * 32];
  cudaMemcpy(out_h, out_d, 128 * 32 * 4, cudaMemcpyDeviceToHost);
  // With A=I(16): D[m][n] = B[n][m] for m<16, 0 for m>=16.
  // Canonical wgmma acc mapping (doc): thread t (of 128), acc[i]:
  //   row = 16*(t/32) + 8*((i%4)/2) + (t%32)/4
  //   col = 8*(i/4) + (t%4)*2 + (i%2)
  printf("LBO16=%d SBO16=%d\n", lbo16, sbo16);
  for (int t = 0; t < 5; ++t) {
    printf("t%-3d:", t);
    for (int i = 0; i < 8; ++i) printf(" %7.0f", out_h[t * 32 + i]);
    printf("\n");
  }
  // expected for t0 acc0..7 under doc mapping: row=(0), cols 0,1, 8,9, ... wait
  // acc0,1 -> col 0,1 row 0: D[0][0]=B[0][0]=0, D[0][1]=B[1][0]=100
  // acc2,3 -> row 8: D[8][0]=B[0][8]=8, D[8][1]=B[1][8]=108
  // acc4,5 -> col 8,9 row 0: D[0][8]=B[8][0]=800, D[0][9]=B[9][0]=900
  printf("expect t0: 0 100 8 108 800 900 808 908\n");
  return 0;
}
