#include <cstdio>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#define TPB 256
// graphExecution reduceFinal: block tree (block barriers) + shfl_down(double) warp reduce.
__global__ void reduceFinal(double* in, double* result, size_t n){
  __shared__ double tmp[TPB];
  cg::thread_block cta = cg::this_thread_block();
  size_t gtid = blockIdx.x*blockDim.x + threadIdx.x;
  double temp_sum = 0.0;
  for (int i=gtid;i<n;i+=gridDim.x*blockDim.x) temp_sum += (double)in[i];
  tmp[cta.thread_rank()] = temp_sum; cg::sync(cta);
  cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);
  if ((blockDim.x>=512)&&(cta.thread_rank()<256)){ tmp[cta.thread_rank()]=temp_sum=temp_sum+tmp[cta.thread_rank()+256]; } cg::sync(cta);
  if ((blockDim.x>=256)&&(cta.thread_rank()<128)){ tmp[cta.thread_rank()]=temp_sum=temp_sum+tmp[cta.thread_rank()+128]; } cg::sync(cta);
  if ((blockDim.x>=128)&&(cta.thread_rank()< 64)){ tmp[cta.thread_rank()]=temp_sum=temp_sum+tmp[cta.thread_rank()+ 64]; } cg::sync(cta);
  if (cta.thread_rank()<32){
    if (blockDim.x>=64) temp_sum += tmp[cta.thread_rank()+32];
    for (int offset=tile32.size()/2; offset>0; offset/=2) temp_sum += tile32.shfl_down(temp_sum, offset);
  }
  if (cta.thread_rank()==0) result[0]=temp_sum;
}
int main(){
  const int N=TPB; double hin[N]; for(int i=0;i<N;i++) hin[i]=(i%7)*0.5+0.001;
  double *din,*dr,hr=0,ref=0; for(int i=0;i<N;i++) ref+=hin[i];
  cudaMalloc(&din,N*8); cudaMalloc(&dr,8);
  cudaMemcpy(din,hin,N*8,cudaMemcpyHostToDevice);
  reduceFinal<<<1,TPB>>>(din,dr,N); cudaDeviceSynchronize();
  cudaMemcpy(&hr,dr,8,cudaMemcpyDeviceToHost);
  printf("result=%.6f ref=%.6f\n",hr,ref);
  cudaFree(din);cudaFree(dr); return 0;
}
