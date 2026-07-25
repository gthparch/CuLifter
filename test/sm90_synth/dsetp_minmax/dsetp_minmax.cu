#include <cstdio>
#include <cuda_runtime.h>
#include <cmath>
// Exercises the f64 min/max idiom that compiles to DSETP.MIN/MAX + FSEL
// (frechet's Frechet-distance min-over-couplings-of-max pattern, with abs).
__global__ void k(const double* a, const double* b, double* outmin, double* outmax, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  double x = fabs(a[i]), y = b[i];
  outmin[i] = fmin(x, y);            // DSETP.MIN + FSEL
  outmax[i] = fmax(x, fabs(y));      // DSETP.MAX + FSEL (abs operands like frechet)
}
int main() {
  const int N = 4096; double *a,*b,*omin,*omax;
  cudaMallocManaged(&a,N*sizeof(double)); cudaMallocManaged(&b,N*sizeof(double));
  cudaMallocManaged(&omin,N*sizeof(double)); cudaMallocManaged(&omax,N*sizeof(double));
  for(int i=0;i<N;i++){ a[i]=((i%101)-50)*0.3125; b[i]=((i%97)-33)*0.5; }
  k<<<(N+255)/256,256>>>(a,b,omin,omax,N);
  cudaDeviceSynchronize();
  double smin=0,smax=0; for(int i=0;i<N;i++){ smin+=omin[i]; smax+=omax[i]; }
  printf("sum_min=%.6f sum_max=%.6f first=(%.4f,%.4f) last=(%.4f,%.4f)\n",
         smin,smax,omin[0],omax[0],omin[N-1],omax[N-1]);
  return 0;
}
