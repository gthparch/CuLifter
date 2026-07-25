#include <cstdio>
#include <cuda_runtime.h>
typedef unsigned long ulong;
__global__ void k(const unsigned* cond, const unsigned* val, ulong* out, int n){
  int t=threadIdx.x; ulong s=0; for(int i=0;i<n;i++){ if(cond[t*n+i]) s+=(ulong)val[t*n+i]; } out[t]=s;
}
int main(){ const int T=32,n=16; unsigned *c,*v; ulong *o; cudaMallocManaged(&c,T*n*4);cudaMallocManaged(&v,T*n*4);cudaMallocManaged(&o,T*8);
  for(int i=0;i<T*n;i++){c[i]=i&1;v[i]=i+1;} ulong rt=0; for(int t=0;t<T;t++){ulong s=0;for(int i=0;i<n;i++)if(c[t*n+i])s+=v[t*n+i];rt+=s;o[t]=0;}
  k<<<1,T>>>(c,v,o,n); cudaDeviceSynchronize(); ulong tt=0;int bad=0;for(int t=0;t<T;t++){tt+=o[t];} printf("tot=%lu ref=%lu %s\n",tt,rt,tt==rt?"PASS":"FAIL"); return 0;}
