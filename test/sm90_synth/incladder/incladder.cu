#include <cstdio>
#include <cuda_runtime.h>
typedef unsigned long ulong;
static __device__ int my_abs(int x){ return x<0?-x:x; }
__global__ void inc_kernel(const ulong* values, unsigned* out) {
  int gid = blockIdx.x*blockDim.x+threadIdx.x;
  ulong value = values[gid];
  unsigned inc;
  if (value == 0) inc = 1;
  else if (my_abs((int)(value)) <= 0xFF) inc = 1;
  else if (my_abs((int)(value)) <= 0xFFFF) inc = 2;
  else if (((value) & 0xFFFF) == 0) inc = 2;
  else if (my_abs((int)((value)&0xFFFF))<=0xFF && my_abs((int)((value>>16)&0xFFFF))<=0xFF) inc = 2;
  else if ((((value)&0xFF)==((value>>8)&0xFF)) && (((value)&0xFF)==((value>>16)&0xFF)) && (((value)&0xFF)==((value>>24)&0xFF))) inc = 1;
  else inc = 4;
  out[gid] = inc;
}
int main(){
  const int WGS=256, BLOCKS=1, N=WGS*BLOCKS;
  ulong *in; unsigned *out;
  cudaMallocManaged(&in,N*sizeof(ulong)); cudaMallocManaged(&out,N*sizeof(unsigned));
  ulong vals[7]={0,5,300,0x10000ul,0x00050007ul,0x07070707ul,0x123456789aul};
  const char* names[7]={"v=0","v=5","v=300","v=0x10000","v=0x50007","v=0x7070707","v=0x123456789a"};
  unsigned exp_inc[7]={1,1,2,2,2,1,4};
  for(int i=0;i<N;i++) in[i]=vals[i%7];
  inc_kernel<<<BLOCKS,WGS>>>(in,out); cudaDeviceSynchronize();
  // print inc for first occurrence of each branch
  int bad=0;
  for(int b=0;b<7;b++){ unsigned got=out[b]; printf("%-14s got_inc=%u exp=%u %s\n",names[b],got,exp_inc[b],got==exp_inc[b]?"ok":"WRONG"); if(got!=exp_inc[b])bad=1;}
  printf("%s\n", bad?"FAIL":"PASS");
  return 0;
}
