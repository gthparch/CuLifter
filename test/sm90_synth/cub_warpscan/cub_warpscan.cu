#include <cstdio>
#include <cub/cub.cuh>
// Isolate CUB WarpScan inclusive-sum (SHFL.UP based) — same primitive family
// thrust::reduce_by_key uses. 32 lanes, all-1 input -> inclusive prefix = lane+1,
// final lane = 32. Tests SHFL.UP scan correctness end to end.
__global__ void wscan(const int* in, int* out){
  typedef cub::WarpScan<int> WarpScan;
  __shared__ typename WarpScan::TempStorage temp;
  int lane = threadIdx.x & 31;
  int v = in[threadIdx.x];
  int agg;
  WarpScan(temp).InclusiveSum(v, v, agg);
  out[threadIdx.x] = v;        // inclusive prefix
  if (lane==0) out[32] = agg;  // warp aggregate
}
int main(){
  const int N=32;
  int hin[N]; for(int i=0;i<N;i++) hin[i]=1;
  int *din,*dout; int hout[N+1];
  cudaMalloc(&din,N*4); cudaMalloc(&dout,(N+1)*4);
  cudaMemcpy(din,hin,N*4,cudaMemcpyHostToDevice);
  wscan<<<1,N>>>(din,dout); cudaDeviceSynchronize();
  cudaMemcpy(hout,dout,(N+1)*4,cudaMemcpyDeviceToHost);
  for(int i=0;i<N;i++) printf("prefix[%d]=%d\n",i,hout[i]);
  printf("aggregate=%d\n",hout[N]);
  cudaFree(din);cudaFree(dout); return 0;
}
