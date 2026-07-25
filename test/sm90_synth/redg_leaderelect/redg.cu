#include <cstdio>
#include <cuda_runtime.h>

// Mirrors hungarian-cuda's step_3 / compress_matrix warp-aggregated global
// atomicAdd idiom. nvcc -O3 turns `if (cond) atomicAdd(&counter, 1)` into a
// warp-aggregated reduction:
//   VOTEU.ANY URmask        ; active-lane ballot
//   UFLO.U32  URleader      ; leader = MSB(mask) = highest active lane
//   POPC      Rn, URmask    ; #active lanes
//   ISETP.EQ  P0, LANEID, URleader
//   @P0 REDG.E.ADD [counter], Rn   ; leader-only adds the popcount
// The leader election (laneid == UFLO(mask)) MUST fire for exactly one lane or
// the counter stays 0 (the hungarian hang: n_matches never increments).
__device__ int counter;

__global__ void count_kernel(const int* pred) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (pred[i] >= 0) {
    atomicAdd(&counter, 1);
  }
}

int main() {
  const int WGS = 256, BLOCKS = 16, N = WGS * BLOCKS;
  int *pred;
  cudaMallocManaged(&pred, N * sizeof(int));
  int expect = 0;
  // Mixed predicate so warps diverge (like row_of_star_at_column[i] >= 0).
  for (int i = 0; i < N; i++) {
    pred[i] = (i % 3 == 0) ? 5 : -1;   // ~1/3 of lanes active
    if (pred[i] >= 0) expect++;
  }
  int zero = 0;
  cudaMemcpyToSymbol(counter, &zero, sizeof(int));
  count_kernel<<<BLOCKS, WGS>>>(pred);
  cudaDeviceSynchronize();
  int h_counter = 0;
  cudaMemcpyFromSymbol(&h_counter, counter, sizeof(int));
  printf("counter=%d expect=%d %s\n", h_counter, expect,
         (h_counter == expect) ? "PASS" : "FAIL");
  return 0;
}
