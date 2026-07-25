#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>
// FULL scatterParticle replica: bounds + struct load + f64 div + scatter (F2I + 2 atomicAdd).
struct Particle { double x; double v; bool alive; };
#define DX 1e-4
#define NI 101
#define THREADS 256
__device__ void scatter(double lc, float value, float* field) {
  int i = (int)lc;
  float di = lc - i;
  atomicAdd(&field[i], value*(1-di));
  atomicAdd(&field[i+1], value*(di));
}
__global__ void scat(const Particle* __restrict particles, float* __restrict den, long N) {
  long p = blockIdx.x*blockDim.x + threadIdx.x;
  if (p < N && particles[p].alive) {
    double lc = (particles[p].x - 0)/DX;
    scatter(lc, 1.f, den);
  }
}
int main(){
  const long N=500000; int nb = 1 + N/THREADS;
  Particle* dp; float* den;
  cudaMallocManaged(&dp, N*sizeof(Particle));
  cudaMallocManaged(&den, NI*sizeof(float));
  for(long i=0;i<N;i++){ dp[i].x=(double)i*(0.01/500000.0); dp[i].v=0; dp[i].alive=true; }
  for(int i=0;i<NI;i++) den[i]=0;
  cudaError_t e0=cudaGetLastError();
  scat<<<nb,THREADS>>>(dp,den,N);
  cudaError_t e=cudaDeviceSynchronize();
  double s=0; for(int i=0;i<NI;i++) s+=den[i];
  printf("launch_err=%s sum_den=%.6g %s\n", cudaGetErrorString(e), s, e==cudaSuccess?"OK":"CRASH");
  return 0;
}
