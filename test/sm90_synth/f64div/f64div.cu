#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>
// Isolate IEEE f64 division a/b (SASS: MUFU.RCP64H + Newton DFMAs + slow path).
__global__ void divk(const double* a, const double* b, double* out, int n){
  int i = blockIdx.x*blockDim.x+threadIdx.x;
  if (i<n) out[i] = a[i] / b[i];
}
int main(){
  const int n=4096; double *a,*b,*o;
  cudaMallocManaged(&a,n*8); cudaMallocManaged(&b,n*8); cudaMallocManaged(&o,n*8);
  for(int i=0;i<n;i++){ a[i]=(double)(i%97)-48.0; b[i]=(double)((i%13)-6)+0.5; }
  divk<<<(n+255)/256,256>>>(a,b,o,n); cudaDeviceSynchronize();
  int bad=0; double maxrel=0;
  for(int i=0;i<n;i++){ double ref=a[i]/b[i]; double e=fabs(o[i]-ref); double rel=e/(fabs(ref)+1e-300); if(rel>maxrel)maxrel=rel; if(rel>1e-14)bad++; }
  printf("maxrel=%g bad=%d %s\n",maxrel,bad,bad==0?"PASS":"FAIL");
  return 0;
}
