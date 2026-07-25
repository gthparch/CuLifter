#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>
__global__ void k(const float* x, const float* y, float* o){
  int t=threadIdx.x;
  o[t] = (fabsf(x[t]) - fabsf(y[t])) / (fabsf(x[t]) + fabsf(y[t]));
}
int main(){ const int T=32; float*x,*y,*o; cudaMallocManaged(&x,T*4);cudaMallocManaged(&y,T*4);cudaMallocManaged(&o,T*4);
  for(int i=0;i<T;i++){ x[i]=(i%2)?-(i+1.0f):(i+1.0f); y[i]=-x[i]; } // x=-y -> |x|+|y|=2|x|
  k<<<1,T>>>(x,y,o); cudaDeviceSynchronize();
  int bad=0; for(int i=0;i<T;i++){ float r=(fabsf(x[i])-fabsf(y[i]))/(fabsf(x[i])+fabsf(y[i])); if(o[i]!=r||isnan(o[i]))bad++; }
  printf("o[0]=%f o[1]=%f bad=%d %s\n",o[0],o[1],bad,bad?"FAIL":"PASS"); return 0;}
