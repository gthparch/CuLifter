#include <cstdio>
#include <cuda_runtime.h>
#include <cmath>
__global__ void k(const double* a, const double* b, double* omin, double* omax, int n) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i>=n) return;
  omin[i] = fmin(a[i], b[i]);
  omax[i] = fmax(a[i], b[i]);
}
int main(){
  const int N=2048; double *a,*b,*omin,*omax;
  cudaMallocManaged(&a,N*sizeof(double)); cudaMallocManaged(&b,N*sizeof(double));
  cudaMallocManaged(&omin,N*sizeof(double)); cudaMallocManaged(&omax,N*sizeof(double));
  for(int i=0;i<N;i++){ a[i]=((i%101)-50)*0.3125; b[i]=((i%97)-33)*0.5; }
  k<<<(N+255)/256,256>>>(a,b,omin,omax,N);
  cudaDeviceSynchronize();
  // print first 8 elements where omin/omax disagree with the host fmin/fmax
  int shown=0;
  for(int i=0;i<N && shown<8;i++){
    double em=fmin(a[i],b[i]), ex=fmax(a[i],b[i]);
    if(omin[i]!=em || omax[i]!=ex){
      printf("MISMATCH i=%d a=%.4f b=%.4f min(got=%.4f exp=%.4f) max(got=%.4f exp=%.4f)\n",
             i,a[i],b[i],omin[i],em,omax[i],ex);
      shown++;
    }
  }
  printf("done shown=%d\n",shown);
  return 0;
}
