#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>
// Mirror atan2's compute_i F2I path (unsafe_atan2i_impl<15>) — the i32 output
// that produces -nan in the lifted nvvm while compute_f (f32) is correct.
__device__ __host__ static int atan2i15(float y, float x){
  long long maxint=(long long)2147483647+1LL; int pi4=int(maxint/4),pi34=int(3LL*maxint/4);
  float r=(fabsf(x)-fabsf(y))/(fabsf(x)+fabsf(y)); if(x<0)r=-r;
  float z=r*r;
  float poly=r*(-683562624.f+z*(227746080.f+z*(-135400128.f+z*(90460848.f+z*(-54431464.f+z*(22973256.f+z*(-4657049.f)))))));
  int angle=(x>=0)?pi4:pi34; angle+=int(poly); return (y<0)?-angle:angle;
}
__global__ void compute_i(int n,const float*x,const float*y,int*r){int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) r[i]=atan2i15(y[i],x[i]);}
int main(){
  const int n=4096; float*x,*y; int*o; cudaMallocManaged(&x,n*4); cudaMallocManaged(&y,n*4); cudaMallocManaged(&o,n*4);
  for(int i=0;i<n;i++){ x[i]=(float)((i%101)-50)+0.3f; y[i]=(float)((i%83)-41)-0.2f; }
  compute_i<<<(n+255)/256,256>>>(n,x,y,o); cudaDeviceSynchronize();
  long long bad=0; for(int i=0;i<n;i++){ int ref=atan2i15(y[i],x[i]); if(o[i]!=ref){bad++; if(bad<=4)printf("  i=%d got=%d ref=%d\n",i,o[i],ref);} }
  printf("bad=%lld %s\n",bad,bad==0?"PASS":"FAIL"); return 0;
}
