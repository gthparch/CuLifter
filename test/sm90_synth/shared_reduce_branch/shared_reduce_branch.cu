#include <cstdio>
#include <cuda_runtime.h>
// inc defined across a multi-branch if-else (PHI) then consumed by shared atomicAdd.
__global__ void k(const unsigned* in, unsigned* cmp_size){
  __shared__ unsigned compressable;
  int lid = threadIdx.x, WGS = blockDim.x;
  int gid = blockIdx.x*WGS+lid;
  unsigned v = in[gid]; unsigned inc;
  if (v == 0) inc = 1;
  else if (v <= 0xFF) inc = 1;
  else if (v <= 0xFFFF) inc = 2;
  else if ((v & 0xFFFF) == 0) inc = 2;
  else if ((v & 0xFF) <= 0xF) inc = 2;
  else if (((v & 0xFF) == ((v>>8)&0xFF))) inc = 1;
  else inc = 4;
  if (lid == 0) compressable = 0;
  __syncthreads();
  atomicAdd(&compressable, inc);
  __syncthreads();
  if (lid == WGS-1) atomicAdd(cmp_size, compressable);
}
int main(){
  const int N=4096; unsigned *din,*dc,hc=0,*hin=new unsigned[N];
  unsigned s=0x12345678u; for(int i=0;i<N;i++){ s=s*1664525u+1013904223u; hin[i]=s>>(i%20); }
  cudaMalloc(&din,N*4); cudaMalloc(&dc,4);
  cudaMemcpy(din,hin,N*4,cudaMemcpyHostToDevice); cudaMemset(dc,0,4);
  k<<<N/256,256>>>(din,dc); cudaDeviceSynchronize();
  cudaMemcpy(&hc,dc,4,cudaMemcpyDeviceToHost);
  printf("cmp_size=%u\n",hc); cudaFree(din);cudaFree(dc);delete[]hin; return 0;
}
