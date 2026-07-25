#include <cstdio>
#include <cuda_runtime.h>
// One block, one warp: __shfl_down_sync tree reduction of 32 lanes.
__global__ void warpsum(const int* in, int* out){
  int lane = threadIdx.x;
  int v = in[lane];
  for (int o = 16; o > 0; o >>= 1)
    v += __shfl_down_sync(0xffffffff, v, o);
  if (lane == 0) out[0] = v;
}
int main(){
  int *in,*out; cudaMallocManaged(&in,32*4); cudaMallocManaged(&out,4);
  int ref=0; for(int i=0;i<32;i++){ in[i]=i*3+1; ref+=in[i]; }
  out[0]=-1;
  warpsum<<<1,32>>>(in,out); cudaDeviceSynchronize();
  printf("sum=%d ref=%d %s\n", out[0], ref, out[0]==ref?"PASS":"FAIL");
  return 0;
}
