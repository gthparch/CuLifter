#include <cstdio>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#define TPB 32
// Dump per-thread temp_sum (after accumulation loop) AND the post-tree tmp[],
// to localize graphExecution's bug: accumulation loop vs tree reduction.
__global__ void reduce(float* in, double* out, double* dbg, size_t n, size_t outSize){
  __shared__ double tmp[TPB];
  cg::thread_block cta = cg::this_thread_block();
  size_t gtid = blockIdx.x*blockDim.x + threadIdx.x;
  double temp_sum = 0.0;
  for (int i=gtid;i<n;i+=gridDim.x*blockDim.x) temp_sum += (double)in[i];
  dbg[gtid] = temp_sum;                 // <-- per-thread accumulation result
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
  float* din; double* dout,*ddbg,hout[BLOCKS],hdbg[N];
  cudaMalloc(&din,N*4);cudaMalloc(&dout,BLOCKS*8);cudaMalloc(&ddbg,N*8);
  cudaMemcpy(din,hin,N*4,cudaMemcpyHostToDevice);
  reduce<<<BLOCKS,TPB>>>(din,dout,ddbg,N,BLOCKS); cudaDeviceSynchronize();
  cudaMemcpy(hout,dout,BLOCKS*8,cudaMemcpyDeviceToHost);
  cudaMemcpy(hdbg,ddbg,N*8,cudaMemcpyDeviceToHost);
  double s=0; for(int i=0;i<N;i++) s+=hdbg[i];
  printf("sum_of_tempsum=%.1f out=%.1f\n",s,hout[0]);
  for(int i=0;i<N;i++) printf("%.0f ",hdbg[i]); printf("\n");
  cudaFree(din);cudaFree(dout);cudaFree(ddbg);delete[]hin; return 0;
}
