#include <cstdio>
#include <cub/cub.cuh>
// CUB WarpReduce (SHFL.DOWN based) — sum reduce 32 lanes of all-1 -> 32 in lane0.
__global__ void wred(const int* in, int* out){
  typedef cub::WarpReduce<int> WarpReduce;
  __shared__ typename WarpReduce::TempStorage temp;
  int lane = threadIdx.x & 31;
  int v = in[threadIdx.x];
  int agg = WarpReduce(temp).Sum(v);
  if (lane==0) out[0] = agg;
}
int main(){
  const int N=32; int hin[N]; for(int i=0;i<N;i++) hin[i]=1;
  int *din,*dout; int hout=0;
  cudaMalloc(&din,N*4); cudaMalloc(&dout,4);
  cudaMemcpy(din,hin,N*4,cudaMemcpyHostToDevice);
  wred<<<1,N>>>(din,dout); cudaDeviceSynchronize();
  cudaMemcpy(&hout,dout,4,cudaMemcpyDeviceToHost);
  printf("warpsum=%d\n",hout);
  cudaFree(din);cudaFree(dout); return 0;
}
