#include <cstdio>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#define TPB 32
// One warp, all-1.0 input -> each of the 32 lanes contributes 1.0; the shared
// double tree reduction should yield 32.0. Isolates the warp tree reduction.
__global__ void reduce(float* in, double* out, size_t n, size_t outSize){
  __shared__ double tmp[TPB];
  cg::thread_block cta = cg::this_thread_block();
  size_t gtid = blockIdx.x*blockDim.x + threadIdx.x;
  double temp_sum = 0.0;
  for (int i=gtid;i<n;i+=gridDim.x*blockDim.x) temp_sum += (double)in[i];
  tmp[cta.thread_rank()] = temp_sum;
  cg::sync(cta);
  cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);
  double beta=temp_sum, temp;
  for (int i=tile32.size()/2;i>0;i>>=1){
    if (tile32.thread_rank()<i){ temp=tmp[cta.thread_rank()+i]; beta+=temp; tmp[cta.thread_rank()]=beta; }
    cg::sync(tile32);
  }
  cg::sync(cta);
  if (cta.thread_rank()==0 && blockIdx.x<outSize){
    beta=0.0; for(int i=0;i<cta.size();i+=tile32.size()) beta+=tmp[i];
    out[blockIdx.x]=beta;
  }
}
int main(){
  const int BLOCKS=1,N=BLOCKS*TPB;
  float* hin=new float[N]; for(int i=0;i<N;i++) hin[i]=1.0f;
  float* din; double* dout,hout[BLOCKS];
  cudaMalloc(&din,N*4);cudaMalloc(&dout,BLOCKS*8);
  cudaMemcpy(din,hin,N*4,cudaMemcpyHostToDevice);
  reduce<<<BLOCKS,TPB>>>(din,dout,N,BLOCKS); cudaDeviceSynchronize();
  cudaMemcpy(hout,dout,BLOCKS*8,cudaMemcpyDeviceToHost);
  for(int b=0;b<BLOCKS;b++) printf("block%d=%.6f\n",b,hout[b]);
  cudaFree(din);cudaFree(dout);delete[]hin; return 0;
}
