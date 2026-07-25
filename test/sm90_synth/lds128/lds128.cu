#include <cstdio>
#include <cuda_runtime.h>
typedef unsigned long ulong;
// thread t writes partial[t]; thread0 reads partial[0..7] as 2x ulong4 (LDS.128) and reduces
__global__ void k(const ulong* in, ulong* out){
  __shared__ ulong partial[8];
  int t = threadIdx.x;
  partial[t] = in[t];
  __syncthreads();
  if (t==0){
    ulong4 a = *(ulong4*)&partial[0];
    ulong4 b = *(ulong4*)&partial[4];
    out[0] = a.x+a.y+a.z+a.w+b.x+b.y+b.z+b.w;
  }
}
int main(){
  ulong *in,*out; cudaMallocManaged(&in,8*8); cudaMallocManaged(&out,8);
  ulong ref=0; for(int i=0;i<8;i++){ in[i]=(ulong)(i+1)*1000000007UL; ref+=in[i]; }
  out[0]=0; k<<<1,8>>>(in,out); cudaDeviceSynchronize();
  printf("sum=%lu ref=%lu %s\n", out[0], ref, out[0]==ref?"PASS":"FAIL");
  return 0;
}
