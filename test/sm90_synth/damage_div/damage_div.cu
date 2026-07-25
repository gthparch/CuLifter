#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
// Replicate damage's exact arithmetic: 1.0 - (double)n / (double)family
// (integer/integer f64 division via RCP64H+Newton). Check bit-exactness.
__global__ void k(const int* n, const int* fam, double* out, int N){
  int i = blockIdx.x*blockDim.x+threadIdx.x;
  if (i<N) out[i] = 1.0 - (double)n[i] / (double)fam[i];
}
int main(){
  const int N=4096; int *n,*fam; double *o;
  cudaMallocManaged(&n,N*4); cudaMallocManaged(&fam,N*4); cudaMallocManaged(&o,N*8);
  for(int i=0;i<N;i++){ fam[i]= (i%60)+3; n[i]= i % (fam[i]+1); }
  k<<<(N+255)/256,256>>>(n,fam,o,N); cudaDeviceSynchronize();
  // bit-exact reference on CPU (same IEEE f64 division)
  int bad=0; double tot=0; uint64_t xо=0;
  for(int i=0;i<N;i++){ double ref=1.0-(double)n[i]/(double)fam[i]; tot+=o[i]; if(o[i]!=ref)bad++; }
  printf("total=%.9f bad=%d %s\n", tot, bad, bad==0?"PASS":"FAIL");
  return 0;
}
