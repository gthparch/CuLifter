#include <cstdio>
#include <cuda_runtime.h>
// Isolate rsqrtf (MUFU.RSQ + predicated range-reduction scaling) — clenergy's
// distances are normal-range (~2.5-6.5) yet lifted energy is ~22% off, pointing
// at the @!P-guarded 2^24/2^12 scaling being mislifted.
__global__ void k(const float* in, float* out, int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
  out[i]=rsqrtf(in[i]);
}
int main(){
  const int N=16; float *in,*out;
  cudaMallocManaged(&in,N*sizeof(float)); cudaMallocManaged(&out,N*sizeof(float));
  float vals[16]={6.5f,2.66f,2.5f,4.41f,1.0f,100.f,0.5f,1e-20f,1e-30f,1e-38f,3.4e38f,16.f,1e10f,1e-10f,2.66f,6.5f};
  for(int i=0;i<N;i++) in[i]=vals[i];
  k<<<1,N>>>(in,out,N);
  cudaDeviceSynchronize();
  for(int i=0;i<8;i++) printf("rsqrt(%.3g)=%.6f ", in[i], out[i]); printf("\n");
  return 0;
}
