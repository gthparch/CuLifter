#include <cstdio>
#include <cuda_runtime.h>
typedef unsigned long ulong;
static __device__ int my_abs(int x){ return x<0?-x:x; }

// fpc_kernel's exact inc-ladder + warp-agg atomic, minimal host.
__global__ void agg_kernel(const ulong* values, unsigned* cmp_size) {
  __shared__ unsigned compressable;
  int lid = threadIdx.x; int WGS = blockDim.x; int gid = blockIdx.x*WGS+lid;
  ulong value = values[gid];
  unsigned inc;
  if (value == 0) inc = 1;
  else if (my_abs((int)(value)) <= 0xFF) inc = 1;
  else if (my_abs((int)(value)) <= 0xFFFF) inc = 2;
  else if (((value) & 0xFFFF) == 0) inc = 2;
  else if (my_abs((int)((value)&0xFFFF))<=0xFF && my_abs((int)((value>>16)&0xFFFF))<=0xFF) inc = 2;
  else if ((((value)&0xFF)==((value>>8)&0xFF)) && (((value)&0xFF)==((value>>16)&0xFF)) && (((value)&0xFF)==((value>>24)&0xFF))) inc = 1;
  else inc = 4;
  if (lid == 0) compressable = 0;
  __syncthreads();
  atomicAdd(&compressable, inc);
  __syncthreads();
  if (lid == WGS-1) atomicAdd(cmp_size, compressable);
}
int main(){
  const int WGS=256, BLOCKS=64, N=WGS*BLOCKS;
  ulong *in; unsigned *out;
  cudaMallocManaged(&in,N*sizeof(ulong)); cudaMallocManaged(&out,sizeof(unsigned)); *out=0;
  // mix of values exercising every branch
  unsigned long long expect=0;
  for(int i=0;i<N;i++){ ulong v; int b=i%7;
    if(b==0)v=0; else if(b==1)v=5; else if(b==2)v=300; else if(b==3)v=0x10000ul; else if(b==4)v=0x00050007ul; else if(b==5)v=0x07070707ul; else v=0x123456789aul;
    in[i]=v; }
  // compute expect on host via same ladder
  for(int i=0;i<N;i++){ ulong value=in[i]; unsigned inc;
    auto ab=[](int x){return x<0?-x:x;};
    if(value==0)inc=1; else if(ab((int)value)<=0xFF)inc=1; else if(ab((int)value)<=0xFFFF)inc=2;
    else if((value&0xFFFF)==0)inc=2; else if(ab((int)(value&0xFFFF))<=0xFF&&ab((int)((value>>16)&0xFFFF))<=0xFF)inc=2;
    else if(((value&0xFF)==((value>>8)&0xFF))&&((value&0xFF)==((value>>16)&0xFF))&&((value&0xFF)==((value>>24)&0xFF)))inc=1; else inc=4;
    expect+=inc; }
  agg_kernel<<<BLOCKS,WGS>>>(in,out); cudaDeviceSynchronize();
  printf("out=%u expect=%llu %s\n",*out,expect,(*out==(unsigned)expect)?"PASS":"FAIL");
  return 0;
}
