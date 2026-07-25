#include <cstdio>
#include <cuda_runtime.h>
typedef unsigned long ulong;
// compute-score essence: bloom-filter bit-test gating a predicated accumulate.
__global__ void k(const unsigned* table, const unsigned* idx, const unsigned* val,
                  ulong* out, int n){
  int t = threadIdx.x;
  ulong sum = 0;
  for (int i = 0; i < n; i++){
    unsigned h = idx[t*n + i];
    bool acc = (table[h >> 5] >> (h & 0x1f)) & 0x1;   // variable-shift bit test
    if (acc) sum += (ulong)val[t*n + i];
  }
  out[t] = sum;
}
int main(){
  const int T=32, n=16;
  unsigned *table,*idx,*val; ulong *out;
  cudaMallocManaged(&table,4096*4); cudaMallocManaged(&idx,T*n*4);
  cudaMallocManaged(&val,T*n*4); cudaMallocManaged(&out,T*8);
  for(int i=0;i<4096;i++) table[i]=0xA5A5A5A5u;            // mixed bits
  for(int i=0;i<T*n;i++){ idx[i]=(i*131+7)%65536; val[i]=i*3+1; }
  ulong ref[T];
  for(int t=0;t<T;t++){ ulong s=0; for(int i=0;i<n;i++){ unsigned h=idx[t*n+i]; if((table[h>>5]>>(h&0x1f))&1) s+=val[t*n+i]; } ref[t]=s; out[t]=0; }
  k<<<1,T>>>(table,idx,val,out,n); cudaDeviceSynchronize();
  int bad=0; ulong tot=0,rtot=0;
  for(int t=0;t<T;t++){ tot+=out[t]; rtot+=ref[t]; if(out[t]!=ref[t])bad++; }
  printf("tot=%lu ref=%lu bad=%d %s\n", tot, rtot, bad, bad==0?"PASS":"FAIL");
  return 0;
}
