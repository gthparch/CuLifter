#include <cstdio>
#include <cuda_runtime.h>
// exact bloom bit-test: gather, then variable-shift the GATHERED value by amt from same source
__global__ void k(const unsigned* table, const unsigned* idx, unsigned* out){
  int t = threadIdx.x;
  unsigned h = idx[t];
  out[t] = (table[h >> 5] >> (h & 0x1f)) & 0x1;
}
int main(){ const int T=32; unsigned *tb,*ix,*o;
  cudaMallocManaged(&tb,2048*4); cudaMallocManaged(&ix,T*4); cudaMallocManaged(&o,T*4);
  for(int i=0;i<2048;i++) tb[i]=0xA5A5A5A5u;
  for(int i=0;i<T;i++) ix[i]=(i*131+7)%65536;
  k<<<1,T>>>(tb,ix,o); cudaDeviceSynchronize();
  int bad=0; for(int i=0;i<T;i++){ unsigned h=(i*131+7)%65536; unsigned r=(0xA5A5A5A5u>>(h&0x1f))&1; if(o[i]!=r)bad++; }
  printf("bad=%d %s\n",bad,bad?"FAIL":"PASS"); return 0; }
