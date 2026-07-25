#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>
// Replicate sheath scatterParticle's prefix EXACTLY: 64-bit bounds, strided
// struct .x load, f64 division by constant DX. Write lc to out[p] so we can
// diff every thread's lc (native vs lifted ptxswap).
struct Particle { double x; double v; bool alive; };
#define DX 1e-4
#define THREADS 256
__global__ void scat(const Particle* __restrict particles, double* __restrict out, long N) {
  long p = blockIdx.x*blockDim.x + threadIdx.x;
  if (p < N && particles[p].alive) {
    double lc = (particles[p].x - 0)/DX;
    out[p] = lc;
  }
}
int main(){
  const long N=500000; int nb = 1 + N/THREADS;
  Particle* dp; double* dout;
  cudaMallocManaged(&dp, N*sizeof(Particle));
  cudaMallocManaged(&dout, (size_t)nb*THREADS*sizeof(double));
  for(long i=0;i<N;i++){ dp[i].x = (double)i*(0.01/500000.0); dp[i].v=0; dp[i].alive=true; }
  for(long i=0;i<(long)nb*THREADS;i++) dout[i]=-999.0;
  scat<<<nb,THREADS>>>(dp,dout,N); cudaDeviceSynchronize();
  int bad=0; long firstbad=-1;
  for(long p=0;p<N;p++){ double ref=(dp[p].x-0)/DX; if(dout[p]!=ref){ if(firstbad<0)firstbad=p; bad++; } }
  if(firstbad>=0) printf("bad=%d first p=%ld lifted_lc=%.10g ref_lc=%.10g x=%.10g FAIL\n",bad,firstbad,dout[firstbad],(dp[firstbad].x-0)/DX,dp[firstbad].x);
  else printf("bad=0 PASS\n");
  return 0;
}
