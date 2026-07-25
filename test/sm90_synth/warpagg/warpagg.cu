#include <cstdio>
#include <cuda_runtime.h>

// Mirrors the fpc warp-aggregated-atomic idiom: every thread atomicAdds a
// uniform-ish value to a shared accumulator, then one thread flushes to global.
// nvcc -O3 warp-aggregates the atomicAdd -> REDUX.SUM + leader-elect + ATOMS.
__global__ void agg_kernel(const unsigned* in, unsigned* out) {
  __shared__ unsigned acc;
  int lid = threadIdx.x;
  int WGS = blockDim.x;
  int gid = blockIdx.x*WGS + lid;
  unsigned inc = in[gid];      // per-lane value (1..) so warp-sum is meaningful
  if (lid == 0) acc = 0;
  __syncthreads();
  atomicAdd(&acc, inc);
  __syncthreads();
  if (lid == WGS-1) atomicAdd(out, acc);
}

int main() {
  const int WGS = 256, BLOCKS = 64, N = WGS*BLOCKS;
  unsigned *in, *out; unsigned h_out=0;
  cudaMallocManaged(&in, N*sizeof(unsigned));
  cudaMallocManaged(&out, sizeof(unsigned));
  *out = 0;
  unsigned long long expect = 0;
  for (int i=0;i<N;i++){ in[i]= (i%7)+1; expect += in[i]; }
  agg_kernel<<<BLOCKS,WGS>>>(in,out);
  cudaDeviceSynchronize();
  h_out = *out;
  printf("out=%u expect=%llu %s\n", h_out, expect, (h_out==expect)?"PASS":"FAIL");
  return 0;
}
