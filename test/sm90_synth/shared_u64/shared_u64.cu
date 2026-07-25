#include <cstdio>
#include <cuda_runtime.h>
typedef unsigned long ulong;
// One block: each thread mulfp-accumulates into __shared__ ulong, syncs, thread0 reduces.
__global__ void redu64(const unsigned* w, const unsigned* f, ulong* out, int n){
  __shared__ ulong partial[128];
  int t = threadIdx.x;
  ulong sum = 0;
  for (int i = t; i < n; i += blockDim.x)
    sum += (ulong)w[i] * (ulong)f[i];   // 32x32->64 widening multiply
  partial[t] = sum;
  __syncthreads();
  if (t == 0) {
    ulong total = 0;
    for (int i = 0; i < blockDim.x; i++) total += partial[i];
    out[0] = total;
  }
}
int main(){
  const int n=1024; unsigned *w,*f; ulong *o;
  cudaMallocManaged(&w,n*4); cudaMallocManaged(&f,n*4); cudaMallocManaged(&o,8);
  ulong ref=0; for(int i=0;i<n;i++){ w[i]=i*7+3; f[i]=i%13+1; ref+=(ulong)w[i]*(ulong)f[i]; }
  o[0]=0;
  redu64<<<1,128>>>(w,f,o,n); cudaDeviceSynchronize();
  printf("sum=%lu ref=%lu %s\n", o[0], ref, o[0]==ref?"PASS":"FAIL");
  return 0;
}
