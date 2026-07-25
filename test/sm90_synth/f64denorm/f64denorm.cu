#include <cstdio>
#include <cuda_runtime.h>
// f64 reciprocal/division with TRUE DENORMAL divisors (< 2.2e-308) — probe
// whether the rcp.approx.ftz.f64 seed flushes denormals to 0 -> inf -> NR
// diverges -> NaN (NaN-cluster hypothesis: gpp/kalman/ccsd).
__global__ void k(const double* d, double* out, int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
  double x=d[i]; out[i]=1.0/x + 7.5/x;
}
int main(){
  const int N=256; double *d,*out;
  cudaMallocManaged(&d,N*sizeof(double)); cudaMallocManaged(&out,N*sizeof(double));
  // mix of true denormals (<2.2e-308) and small normals
  double vals[8]={1e-310, 5e-309, 1e-308, 2.5e-308, 4.9e-324/*min denorm*/, 1e-320, -3e-310, 1e-7};
  for(int i=0;i<N;i++) d[i]=vals[i%8];
  k<<<(N+255)/256,256>>>(d,out,N);
  cudaDeviceSynchronize();
  int nans=0,infs=0; for(int i=0;i<8;i++){ if(out[i]!=out[i])nans++; else if(out[i]>1e307||out[i]<-1e307)infs++; }
  printf("first8: "); for(int i=0;i<8;i++) printf("%.4e ", out[i]); printf("\nnans=%d infs=%d\n",nans,infs);
  return 0;
}
