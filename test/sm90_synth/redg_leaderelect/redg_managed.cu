#include <cstdio>
#include <cuda_runtime.h>

// Same warp-aggregated global atomicAdd idiom, but the counter is __managed__
// and read DIRECTLY by the host (no cudaMemcpyFromSymbol) — exactly like
// hungarian-cuda's n_matches / zeros_size. Tests the ptxswap launch-boundary
// device-global sync against managed unified memory.
__managed__ __device__ int counter;

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
  for (int i = 0; i < N; i++) {
    pred[i] = (i % 3 == 0) ? 5 : -1;
    if (pred[i] >= 0) expect++;
  }
  counter = 0;                         // direct managed write
  count_kernel<<<BLOCKS, WGS>>>(pred);
  cudaDeviceSynchronize();
  int h_counter = counter;             // direct managed read
  printf("counter=%d expect=%d %s\n", h_counter, expect,
         (h_counter == expect) ? "PASS" : "FAIL");
  return 0;
}
