#include <cstdio>
#include <cuda_runtime.h>
// f32<->f64 conversions (F2F.F64.F32 fpext, F2F.F32.F64 fptrunc), LBM-style mix
__global__ void k(const float* fin, const double* din, double* dout, float* fout){
  int t = threadIdx.x;
  double d = (double)fin[t];          // F2F.F64.F32 (fpext)
  d = d * 2.5 + 1.0;
  dout[t] = d;
  fout[t] = (float)(din[t] * 0.5);    // F2F.F32.F64 (fptrunc)
}
int main(){ const int T=32; float *fi,*fo; double *di,*doo;
  cudaMallocManaged(&fi,T*4);cudaMallocManaged(&fo,T*4);cudaMallocManaged(&di,T*8);cudaMallocManaged(&doo,T*8);
  for(int i=0;i<T;i++){fi[i]=i*1.5f+0.25f;di[i]=i*3.0+0.125;}
  k<<<1,T>>>(fi,di,doo,fo); cudaDeviceSynchronize();
  int bad=0; for(int i=0;i<T;i++){ double rd=(double)(i*1.5f+0.25f)*2.5+1.0; float rf=(float)((i*3.0+0.125)*0.5);
    if(doo[i]!=rd||fo[i]!=rf)bad++; }
  printf("bad=%d %s\n",bad,bad?"FAIL":"PASS"); return 0; }
