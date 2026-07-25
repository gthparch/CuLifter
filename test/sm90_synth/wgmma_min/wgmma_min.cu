// wgmma_min — minimal Hopper warpgroup-MMA microbench.
// One CTA of 128 threads (one warpgroup) computes C[64x64] += A[64x16]*B[16x64]
// with a single wgmma.mma_async.m64n64k16.f32.bf16.bf16, both operands from
// shared memory via GMMA descriptors (the FA3 SS form -> SASS
// HGMMA.64x64x16.F32.BF16 Racc, gdesc[URn], Racc). Deterministic inputs,
// prints an output checksum + probes; vanilla-vs-lifted must match bitwise.
#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

__device__ __forceinline__ unsigned smem_u32(const void* p) {
  return (unsigned)__cvta_generic_to_shared(p);
}

// GMMA shared-memory matrix descriptor (PTX ISA "asynchronous warpgroup
// matrix descriptor", no-swizzle layout): base>>4 in bits 0..13,
// leading-dim byte offset >>4 in 16..29, stride-dim byte offset >>4 in 32..45.
__device__ __forceinline__ unsigned long long make_desc(unsigned base,
                                                        unsigned lbo,
                                                        unsigned sbo) {
  // lbo/sbo already in 16-byte units
  return ((unsigned long long)(base >> 4) & 0x3FFFull)
       | (((unsigned long long)lbo & 0x3FFFull) << 16)
       | (((unsigned long long)sbo & 0x3FFFull) << 32);
}

__global__ void wgmma_kernel(const __nv_bfloat16* __restrict__ A,
                             const __nv_bfloat16* __restrict__ B,
                             float* __restrict__ C) {
  // K-major (row-major 64x16 A: core matrix 8x8 tiles) no-swizzle layout:
  // A is 64 rows x 16 cols bf16; B is 64 cols x 16 rows (n-major) bf16.
  __shared__ alignas(128) __nv_bfloat16 sA[64 * 16];
  __shared__ alignas(128) __nv_bfloat16 sB[64 * 16];
  const int tid = threadIdx.x;

  for (int idx = tid; idx < 64 * 16; idx += blockDim.x) {
    int i = idx / 16, k = idx % 16;
    // cute Layout_K_INTER canonical: (i/8)*64 + (i%8)*8 + (k%8) + (k/8)*512
    int off = (i / 8) * 64 + (i % 8) * 8 + (k % 8) + (k / 8) * 512;
    sA[off] = A[i * 16 + k];
    sB[off] = B[i * 16 + k];
  }
  __syncthreads();
  asm volatile("fence.proxy.async.shared::cta;");

  // Descriptors (no swizzle): LBO = bytes between 8x8 core-matrix rows
  // = 16 bf16 * 8 = 256B? Use the canonical K-major encoding: LBO=16B rows...
  // For the no-swizzle 64x16 K-major layout cute uses LBO=256, SBO=512.
  // verified on H100 (probe_elem): K-INTER no-swizzle -> LBO16=64, SBO16=8
  unsigned long long dA = make_desc(smem_u32(sA), 64, 8);
  unsigned long long dB = make_desc(smem_u32(sB), 64, 8);

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

  // wgmma m64n64 f32 accumulator layout: thread t of the warpgroup owns
  // 32 elements; store them in the canonical (row,col) mapping.
  for (int i = 0; i < 32; ++i) {
    int r = 16 * (tid / 32) + 8 * ((i % 4) / 2) + (tid % 32) / 4;
    int c = 8 * (i / 4) + (tid % 4) * 2 + (i % 2);
    C[r * 64 + c] = acc[i];
  }
}

int main() {
  const int MK = 64 * 16, NK = 64 * 16, MN = 64 * 64;
  __nv_bfloat16 *A_h = new __nv_bfloat16[MK], *B_h = new __nv_bfloat16[NK];
  // K-major shared layout == the linear order we copy; fill deterministically.
  for (int i = 0; i < MK; ++i) A_h[i] = __float2bfloat16((float)((i * 13 + 7) % 17) * 0.125f);
  for (int i = 0; i < NK; ++i) B_h[i] = __float2bfloat16((float)((i * 5 + 3) % 19) * 0.0625f);
  __nv_bfloat16 *A_d, *B_d; float* C_d;
  cudaMalloc(&A_d, MK * 2); cudaMalloc(&B_d, NK * 2); cudaMalloc(&C_d, MN * 4);
  cudaMemcpy(A_d, A_h, MK * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, B_h, NK * 2, cudaMemcpyHostToDevice);
  wgmma_kernel<<<1, 128>>>(A_d, B_d, C_d);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) { printf("CUDA ERROR: %s\n", cudaGetErrorString(err)); return 1; }
  float* C_h = new float[MN];
  cudaMemcpy(C_h, C_d, MN * 4, cudaMemcpyDeviceToHost);
  double sum = 0; for (int i = 0; i < MN; ++i) sum += C_h[i];
  printf("checksum %.6f\n", sum);
  printf("C[0]=%.4f C[63]=%.4f C[4032]=%.4f C[4095]=%.4f\n",
         C_h[0], C_h[63], C_h[63 * 64], C_h[4095]);
  for (int r = 0; r < 4; ++r) {
    printf("row%d:", r);
    for (int c = 0; c < 8; ++c) printf(" %.3f", C_h[r * 64 + c]);
    printf("\n");
  }
  printf("row8:");
  for (int c = 0; c < 8; ++c) printf(" %.3f", C_h[8 * 64 + c]);
  printf("\nrow16:");
  for (int c = 0; c < 8; ++c) printf(" %.3f", C_h[16 * 64 + c]);
  printf("\n");
  return 0;
}
