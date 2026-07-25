#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>
// Isolate IEEE f32 division a/b (SASS: MUFU.RCP + Newton + FCHK slow-path).
__global__ void divk(const float* a, const float* b, float* out, int n){
  int i = blockIdx.x*blockDim.x+threadIdx.x;
  if (i<n) out[i] = a[i] / b[i];
}
int main(){
  const int n=4096; float *a,*b,*o;
  cudaMallocManaged(&a,n*4); cudaMallocManaged(&b,n*4); cudaMallocManaged(&o,n*4);
  for(int i=0;i<n;i++){ a[i]=(float)(i%97)-48.0f; b[i]=(float)((i%13)-6)+0.5f; }
  divk<<<(n+255)/256,256>>>(a,b,o,n); cudaDeviceSynchronize();
  int bad=0; double maxerr=0;
  for(int i=0;i<n;i++){ float ref=a[i]/b[i]; double e=fabs((double)o[i]-(double)ref); if(e>maxerr)maxerr=e; if(!(e<=1e-6*fabs((double)ref)+1e-7)){bad++;} }
  printf("maxerr=%g bad=%d %s\n",maxerr,bad,bad==0?"PASS":"FAIL");
  return 0;
}
