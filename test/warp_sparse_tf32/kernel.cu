#include <stdint.h>

template <int Selector>
__device__ void issue_tf32(unsigned const *a, unsigned const *b, unsigned metadata,
                           float const *c, float *d) {
  asm volatile("mma.sp.sync.aligned.m16n8k16.row.col.f32.tf32.tf32.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, %16, %17;"
               : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
                 "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]),
                 "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]), "r"(metadata), "n"(Selector));
}

extern "C" __global__ void kern_hmma_sp16816tf32(const uint32_t *a, const uint32_t *b,
                                                const uint32_t *metadata, const float *c, float *d) {
  const unsigned lane = threadIdx.x;
  uint32_t aFragment[4], bFragment[4];
  float cFragment[4], d0[4], d1[4];
#pragma unroll
  for (unsigned r = 0; r < 4; ++r) {
    aFragment[r] = a[4 * lane + r];
    bFragment[r] = b[4 * lane + r];
    cFragment[r] = c[4 * lane + r];
  }
  issue_tf32<0>(aFragment, bFragment, metadata[lane], cFragment, d0);
  issue_tf32<1>(aFragment, bFragment, metadata[32 + lane], cFragment, d1);
#pragma unroll
  for (unsigned r = 0; r < 4; ++r) {
    d[4 * lane + r] = d0[r];
    d[128 + 4 * lane + r] = d1[r];
  }
}
