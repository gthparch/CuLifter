#include <stdint.h>

#define SPARSE_HALF_MMA(TYPE, DATA, KERNEL) \
template <int Selector> \
__device__ void DATA(const uint32_t *a, const uint32_t *b, uint32_t metadata, const float *c, float *d) { \
  asm volatile("mma.sp.sync.aligned.m16n8k32.row.col.f32." TYPE "." TYPE ".f32 " \
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, %16, %17;" \
               : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]) \
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]), \
                 "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]), "r"(metadata), "n"(Selector)); \
} \
extern "C" __global__ void KERNEL(const uint32_t *a, const uint32_t *b, const uint32_t *metadata, \
                                   const float *c, float *d) { \
  const unsigned lane = threadIdx.x; \
  DATA<0>(a + 4 * lane, b + 4 * lane, metadata[lane], c + 4 * lane, d + 4 * lane); \
  DATA<1>(a + 4 * lane, b + 4 * lane, metadata[32 + lane], c + 4 * lane, d + 128 + 4 * lane); \
}

SPARSE_HALF_MMA("f16", issue_fp16, kern_hmma_sp16832f16)
SPARSE_HALF_MMA("bf16", issue_bf16, kern_hmma_sp16832bf16)
#undef SPARSE_HALF_MMA
