#include <cstdio>
#include <cuda_runtime.h>
// f64 reciprocal/division across edge values (tiny, large, negative, near-zero)
// — probe whether MUFU.RCP64H + Newton-Raphson diverges (NaN cluster: kalman/
// gpp/ccsd produce -nan from finite f64 division).
__global__ void k(const double* d, double* out, int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=n) return;
  double x=d[i];
  out[i] = 1.0/x + 3.0/(x*x+1e-300) - x/(x+1e-12);
}
int main(){
  const int N=4096; double *d,*out;
  cudaMallocManaged(&d,N*sizeof(double)); cudaMallocManaged(&out,N*sizeof(double));
  for(int i=0;i<N;i++){
    int m=i%8;
    d[i] = m==0?1e-300 : m==1?1e300 : m==2?-1e-12 : m==3?1e-12
          : m==4?-3.5 : m==5?123456.789 : m==6?1e-8 : 2.0+i*0.001;
  }
  k<<<(N+255)/256,256>>>(d,out,N);
  cudaDeviceSynchronize();
  double s=0; int nans=0;
  for(int i=0;i<N;i++){ if(out[i]==out[i]) s+=out[i]; else nans++; }
  printf("sum=%.6e nans=%d first4=%.4e %.4e %.4e %.4e\n",s,nans,out[0],out[1],out[2],out[3]);
  return 0;
}
