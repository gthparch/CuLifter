#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
typedef unsigned long ulong;
__host__ __device__ unsigned my_abs(int x){ unsigned t = x >> 31; return (x ^ t) - t; }
__global__ void fpc_kernel(const ulong* values, unsigned* cmp_size, unsigned char* incs){
  int gid = blockIdx.x*blockDim.x+threadIdx.x;
  ulong value = values[gid]; unsigned inc;
  if (value == 0) inc = 1;
  else if (my_abs((int)(value)) <= 0xFF) inc = 1;
  else if (my_abs((int)(value)) <= 0xFFFF) inc = 2;
  else if (((value) & 0xFFFF) == 0) inc = 2;
  else if (my_abs((int)((value) & 0xFFFF)) <= 0xFF
        && my_abs((int)((value >> 16) & 0xFFFF)) <= 0xFF) inc = 2;
  else if ((((value) & 0xFF) == ((value >> 8) & 0xFF)) &&
           (((value) & 0xFF) == ((value >> 16) & 0xFF)) &&
           (((value) & 0xFF) == ((value >> 24) & 0xFF))) inc = 1;
  else inc = 4;
  incs[gid] = (unsigned char)inc;
  atomicAdd(cmp_size, inc);
}
int main(){
  const int N=256; ulong hv[N];
  uint64_t s=0x12345678ULL;
  for(int i=0;i<N;i++){ s=s*6364136223846793005ULL+1442695040888963407ULL;
    switch(i%8){ case 0:hv[i]=0;break; case 1:hv[i]=(uint64_t)(int64_t)(int8_t)(s&0xff);break;
      case 2:hv[i]=(s&0xFFFF);break; case 3:hv[i]=(s&0xFFFFFFFFFFFF0000ULL);break;
      case 4:{uint64_t b=s&0xFF; hv[i]=b|(b<<8)|(b<<16)|(b<<24);}break; default:hv[i]=s;break; } }
  ulong*dv; unsigned*dc,hc=0; unsigned char*di,hi[N];
  cudaMalloc(&dv,N*8);cudaMalloc(&dc,4);cudaMalloc(&di,N);
  cudaMemcpy(dv,hv,N*8,cudaMemcpyHostToDevice); cudaMemset(dc,0,4);
  fpc_kernel<<<1,N>>>(dv,dc,di); cudaDeviceSynchronize();
  cudaMemcpy(&hc,dc,4,cudaMemcpyDeviceToHost); cudaMemcpy(hi,di,N,cudaMemcpyDeviceToHost);
  printf("cmp_size=%u\n",hc);
  for(int i=0;i<N;i++) printf("%d%s",hi[i],(i%32==31)?"\n":" ");
  cudaFree(dv);cudaFree(dc);cudaFree(di); return 0;
}
