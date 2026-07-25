#include <cstdio>
#include <cuda_runtime.h>
// contended atomicAdd to SHARED mem -> nvcc warp-aggregates via ATOMS
__global__ void k(const int* cond, const int* val, int* out, int n){
  __shared__ int acc;
  int t = threadIdx.x;
  if (t == 0) acc = 0;
  __syncthreads();
  for (int i = 0; i < n; i++)
    if (cond[t*n+i]) atomicAdd(&acc, val[t*n+i]);
  __syncthreads();
  if (t == 0) out[0] = acc;
}
int main(){
  const int T=64, n=16; int *c,*v,*o; cudaMallocManaged(&c,T*n*4);cudaMallocManaged(&v,T*n*4);cudaMallocManaged(&o,4);
  long ref=0; for(int i=0;i<T*n;i++){ c[i]=(i*7+3)%5!=0; v[i]=(i%17)+1; if(c[i])ref+=v[i]; }
  o[0]=0; k<<<1,T>>>(c,v,o,n); cudaDeviceSynchronize();
  printf("sum=%d ref=%ld %s\n", o[0], ref, o[0]==ref?"PASS":"FAIL"); return 0; }
