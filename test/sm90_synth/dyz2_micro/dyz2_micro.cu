#include <cstdio>
#include <cuda_runtime.h>
#define UNROLLX 8
#define BLOCKSIZEX 8
// clenergy distance compute WITHOUT rsqrt/unroll/accumulate — output dyz2, dx, dist2
__global__ void k(const float gridspacing, float *out, const float4 *atominfo, int natoms){
  unsigned int xindex = blockIdx.x*blockDim.x*UNROLLX + threadIdx.x;
  unsigned int yindex = blockIdx.y*blockDim.y + threadIdx.y;
  unsigned int outaddr = gridDim.x*blockDim.x*UNROLLX*yindex + xindex;
  float coory = gridspacing*yindex, coorx = gridspacing*xindex;
  float dy = coory - atominfo[0].y;
  float dyz2 = dy*dy + atominfo[0].z;
  float dx = coorx - atominfo[0].x; out[outaddr] = rsqrtf(dx*dx+dyz2);
}
int main(){
  float4 *atom; float *out;
  cudaMallocManaged(&atom, sizeof(float4)); cudaMallocManaged(&out, 100000*sizeof(float));
  for(int i=0;i<100000;i++) out[i]=0;
  atom[0]=make_float4(2.0f,1.5f,0.25f,-1.0f);
  dim3 b(8,8), g(8,8);
  k<<<g,b>>>(0.1f, out, atom, 1);
  cudaDeviceSynchronize();
  printf("rsq[0,0]=%.6f rsq[1,0]=%.6f rsq512=%.6f\n", out[0], out[1], out[512]);
  return 0;
}
