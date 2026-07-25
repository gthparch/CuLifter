#include <cstdio>
#include <cuda_runtime.h>
// Exact fpc accumulation pattern: shared atomicAdd + lid==WGS-1 flushes to global.
__global__ void k(unsigned* cmp_size){
  __shared__ unsigned compressable;
  int lid = threadIdx.x, WGS = blockDim.x;
  int gid = blockIdx.x*WGS+lid;
  unsigned inc = (gid % 4) + 1;
  if (lid == 0) compressable = 0;
  __syncthreads();
  atomicAdd(&compressable, inc);
  __syncthreads();
  if (lid == WGS-1) atomicAdd(cmp_size, compressable);
}
int main(){
  const int N=4096; unsigned *dc, hc=0;
  cudaMalloc(&dc,4); cudaMemset(dc,0,4);
  k<<<N/256,256>>>(dc); cudaDeviceSynchronize();
  cudaMemcpy(&hc,dc,4,cudaMemcpyDeviceToHost);
  printf("cmp_size=%u\n",hc);
  cudaFree(dc); return 0;
}
