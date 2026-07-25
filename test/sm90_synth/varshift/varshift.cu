#include <cstdio>
#include <cuda_runtime.h>
__global__ void k(const unsigned* v, const unsigned* amt, unsigned* out){
  int t=threadIdx.x; out[t]=(v[t] >> (amt[t] & 0x1f)) & 0x1;
}
int main(){ const int T=32; unsigned *v,*a,*o; cudaMallocManaged(&v,T*4);cudaMallocManaged(&a,T*4);cudaMallocManaged(&o,T*4);
  for(int i=0;i<T;i++){v[i]=0xA5A5A5A5u;a[i]=i;} k<<<1,T>>>(v,a,o); cudaDeviceSynchronize();
  int bad=0; for(int i=0;i<T;i++){unsigned r=(0xA5A5A5A5u>>(i&31))&1; if(o[i]!=r)bad++;} printf("bad=%d %s\n",bad,bad?"FAIL":"PASS"); return 0;}
