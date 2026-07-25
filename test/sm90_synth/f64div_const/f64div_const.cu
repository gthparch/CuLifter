#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>
// Isolate sheath's XtoL: f64 division by a COMPILE-TIME CONSTANT (DX=1e-4).
// SASS: MUFU.RCP64H R3, <imm DX> + Newton DFMAs + FCHK-guarded slow-path CALL.
#define DX 1e-4
__global__ void divk(const double* a, double* out, int n){
  int i = blockIdx.x*blockDim.x+threadIdx.x;
  if (i<n) out[i] = (a[i] - 0) / DX;   // exact XtoL form
}
int main(){
  const int n=500000; double *a,*o;
  cudaMallocManaged(&a,n*8); cudaMallocManaged(&o,n*8);
  for(int i=0;i<n;i++){ a[i]= (double)i * (0.01/500000.0); } // sheath particle x in [0,0.01)
  divk<<<(n+255)/256,256>>>(a,o,n); cudaDeviceSynchronize();
  int bad=0,neg=0; double maxrel=0;
  for(int i=0;i<n;i++){ double ref=(a[i]-0)/DX; double e=fabs(o[i]-ref); double rel=e/(fabs(ref)+1e-300); if(rel>maxrel)maxrel=rel; if(rel>1e-13)bad++; if(o[i]<0)neg++; }
  printf("maxrel=%g bad=%d neg=%d %s\n",maxrel,bad,neg,(bad==0&&neg==0)?"PASS":"FAIL");
  return 0;
}
