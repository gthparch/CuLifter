#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
typedef unsigned long ulong;
__host__ __device__ unsigned my_abs(int x){ unsigned t = x >> 31; return (x ^ t) - t; }
// Output the branch index (0..6) chosen per value, to pinpoint the divergent condition.
__global__ void cls(const ulong* values, unsigned char* out){
  int gid = blockIdx.x*blockDim.x+threadIdx.x;
  ulong value = values[gid]; unsigned char b;
  if (value == 0) b=0;
  else if (my_abs((int)(value)) <= 0xFF) b=1;
  else if (my_abs((int)(value)) <= 0xFFFF) b=2;
  else if (((value) & 0xFFFF) == 0) b=3;
  else if (my_abs((int)((value) & 0xFFFF)) <= 0xFF
        && my_abs((int)((value >> 16) & 0xFFFF)) <= 0xFF) b=4;
  else if ((((value) & 0xFF) == ((value >> 8) & 0xFF)) &&
           (((value) & 0xFF) == ((value >> 16) & 0xFF)) &&
           (((value) & 0xFF) == ((value >> 24) & 0xFF))) b=5;
  else b=6;
  out[gid]=b;
}
int main(){
  const int N=4096; ulong hv[N];
  uint64_t s=0x12345678ULL;
  for(int i=0;i<N;i++){ s=s*6364136223846793005ULL+1442695040888963407ULL;
    switch(i%8){ case 0:hv[i]=0;break; case 1:hv[i]=(uint64_t)(int64_t)(int8_t)(s&0xff);break;
      case 2:hv[i]=(s&0xFFFF);break; case 3:hv[i]=(s&0xFFFFFFFFFFFF0000ULL);break;
      case 4:{uint64_t b=s&0xFF; hv[i]=b|(b<<8)|(b<<16)|(b<<24);}break; default:hv[i]=s;break; } }
  ulong*dv; unsigned char*dout,hout[N];
  cudaMalloc(&dv,N*8);cudaMalloc(&dout,N);
  cudaMemcpy(dv,hv,N*8,cudaMemcpyHostToDevice);
  cls<<<N/256,256>>>(dv,dout); cudaDeviceSynchronize();
  cudaMemcpy(hout,dout,N,cudaMemcpyDeviceToHost);
  for(int i=0;i<N;i++) printf("%d%s",hout[i],(i%32==31)?"\n":" ");
  cudaFree(dv);cudaFree(dout); return 0;
}
