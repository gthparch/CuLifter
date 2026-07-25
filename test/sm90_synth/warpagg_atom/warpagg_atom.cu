#include <cstdio>
#include <cuda_runtime.h>
// data-dependent atomicAdd to a contended address -> nvcc warp-aggregates it (VOTEU+sum+leader atom)
__global__ void k(const int* cond, const int* val, int* out, int n){
  int t = threadIdx.x;
  for (int i = 0; i < n; i++)
    if (cond[t*n+i]) atomicAdd(out, val[t*n+i]);
}
int main(){
  const int T=64, n=16; int *c,*v,*o; cudaMallocManaged(&c,T*n*4);cudaMallocManaged(&v,T*n*4);cudaMallocManaged(&o,4);
  long ref=0; for(int i=0;i<T*n;i++){ c[i]=(i*7+3)%5!=0; v[i]=(i%17)+1; if(c[i])ref+=v[i]; }
  o[0]=0; k<<<1,T>>>(c,v,o,n); cudaDeviceSynchronize();
  printf("sum=%d ref=%ld %s\n", o[0], ref, o[0]==ref?"PASS":"FAIL"); return 0; }
