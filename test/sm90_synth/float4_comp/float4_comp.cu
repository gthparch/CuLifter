#include <cstdio>
#include <cuda_runtime.h>
// Load a float4 (LDG.E.128) and use each component DISTINCTLY — catches a
// scrambled component-to-register assignment that a load->store roundtrip
// (uint4ldst) is blind to. clenergy reads atom.x/.y/.z/.w in different arith.
__global__ void k(const float4* in, float* out, int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
  float4 v = in[i];
  out[i] = v.x*1.f + v.y*10.f + v.z*100.f + v.w*1000.f;
}
int main(){
  const int N=256; float4 *in; float *out;
  cudaMallocManaged(&in,N*sizeof(float4)); cudaMallocManaged(&out,N*sizeof(float));
  for(int i=0;i<N;i++) in[i]=make_float4(2.f,1.5f,0.25f,-1.f);
  k<<<(N+255)/256,256>>>(in,out,N);
  cudaDeviceSynchronize();
  // expected: 2*1 + 1.5*10 + 0.25*100 + (-1)*1000 = 2+15+25-1000 = -958
  printf("out[0]=%.4f expect=-958.0000\n", out[0]);
  return 0;
}
