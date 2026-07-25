#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
typedef unsigned long ulong;

__host__ __device__ unsigned my_abs(int x){ unsigned t = x >> 31; return (x ^ t) - t; }

// Exact fpc_kernel classification + atomic accumulation.
__global__ void fpc_kernel(const ulong* values, unsigned* cmp_size){
  
  int lid = threadIdx.x, WGS = blockDim.x;
  int gid = blockIdx.x*WGS+lid;
  ulong value = values[gid];
  unsigned inc;
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
  
  atomicAdd(cmp_size, inc);
  
}

int main(){
  const int N = 4096;
  ulong* hv = new ulong[N];
  // diverse values exercising every classification branch
  uint64_t s = 0x12345678ULL;
  for(int i=0;i<N;i++){ s = s*6364136223846793005ULL + 1442695040888963407ULL;
    int cls = i % 8;
    switch(cls){
      case 0: hv[i]=0; break;
      case 1: hv[i]=(uint64_t)(int64_t)(int8_t)(s&0xff); break;       // small signed
      case 2: hv[i]=(s & 0xFFFF); break;                              // 16-bit
      case 3: hv[i]=(s & 0xFFFFFFFFFFFF0000ULL); break;               // low16 zero
      case 4: hv[i]=(s & 0xFF) | ((s&0xFF)<<8)|((s&0xFF)<<16)|((s&0xFF)<<24); break;
      default: hv[i]=s; break;                                        // full 64-bit
    }
  }
  ulong* dv; unsigned* dc, hc=0;
  cudaMalloc(&dv,N*8); cudaMalloc(&dc,4);
  cudaMemcpy(dv,hv,N*8,cudaMemcpyHostToDevice);
  cudaMemset(dc,0,4);
  fpc_kernel<<<N/256,256>>>(dv,dc);
  cudaDeviceSynchronize();
  cudaMemcpy(&hc,dc,4,cudaMemcpyDeviceToHost);
  printf("cmp_size=%u\n",hc);
  cudaFree(dv);cudaFree(dc); delete[] hv; return 0;
}
