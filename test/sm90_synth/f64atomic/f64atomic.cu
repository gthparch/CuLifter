#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>
// Minimal REDG.E.ADD.F64 reproduction: every thread atomicAdds a distinct double
// to one global accumulator. Lifter must pack R5:R4 -> double (not fpext R4).
__global__ void addk(double* acc, const double* v, int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) atomicAdd(acc, v[i]);
}
int main(){
  const int n=4096; double *acc,*v; cudaMallocManaged(&acc,8); cudaMallocManaged(&v,n*8); *acc=0.0;
  double expect=0; for(int i=0;i<n;i++){ v[i]=1.0+i*0.5; expect+=v[i]; }
  addk<<<(n+255)/256,256>>>(acc,v,n); cudaDeviceSynchronize();
  double got=*acc; printf("got=%.4f expect=%.4f %s\n",got,expect, fabs(got-expect)<1e-3?"PASS":"FAIL");
  return 0;
}
