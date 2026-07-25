#include <cstdio>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
// __half2 atomicAdd via CAS spin loop (ATOMS.CAST.SPIN) — like f16atomic
__global__ void k(__half2* r){
  __half2 val = __halves2half2(__float2half(0.f), __float2half(1.f));
  atomicAdd(&r[threadIdx.x % 4], val);
}
int main(){
  __half2* r; cudaMallocManaged(&r,4*sizeof(__half2));
  for(int i=0;i<4;i++) r[i]=__halves2half2(__float2half(0.f),__float2half(0.f));
  k<<<1,256>>>(r); cudaDeviceSynchronize();
  float lo=__low2float(r[0]), hi=__high2float(r[0]);
  printf("r[0]=(%.1f,%.1f) expect (0,64) %s\n", lo, hi, (lo==0.f&&hi==64.f)?"PASS":"FAIL");
  return 0; }
