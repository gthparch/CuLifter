#include <cstdio>
#include <cuda_runtime.h>
// Replicate damage's dynamic-shared tree reduction exactly.
__global__ void reduce(const int* in, int* out, int n){
  extern __shared__ int cache[];
  int lid = threadIdx.x, ls = blockDim.x, gid = blockIdx.x*ls + lid;
  if (gid >= n) return;
  cache[lid] = in[gid];
  __syncthreads();
  for (int i = ls/2; i > 0; i /= 2){
    if (lid < i) cache[lid] += cache[lid+i];
    __syncthreads();
  }
  if (lid == 0) out[blockIdx.x] = cache[0];
}
int main(){
  const int BS=256, NB=4096, n=BS*NB;
  int *in,*out; cudaMallocManaged(&in,n*4); cudaMallocManaged(&out,NB*4);
  for(int i=0;i<n;i++) in[i]= (i*1103515245+12345) & 1;  // deterministic 0/1
  long ref=0; for(int i=0;i<n;i++) ref+=in[i];
  reduce<<<NB,BS,BS*sizeof(int)>>>(in,out,n); cudaDeviceSynchronize();
  long tot=0; for(int b=0;b<NB;b++) tot+=out[b];
  printf("total=%ld ref=%ld %s\n", tot, ref, tot==ref?"PASS":"FAIL");
  return 0;
}
