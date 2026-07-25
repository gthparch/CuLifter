#include <cstdio>
#include <cuda_runtime.h>
// Exact clenergy cenergy() kernel, reduced to 1 atom + 1 block (8x8x8 unroll)
// to isolate the per-grid-point divergence (coords/dx-unroll/rsqrt/accumulate).
#define UNROLLX 1
#define BLOCKSIZEX 8
__global__ void cenergy(const int numatoms, const float gridspacing,
                        float *energygrid, const float4 *atominfo) {
  unsigned int xindex  = blockIdx.x * blockDim.x * UNROLLX + threadIdx.x;
  unsigned int yindex  = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int outaddr = gridDim.x * blockDim.x * UNROLLX * yindex + xindex;
  float coory = gridspacing * yindex;
  float coorx = gridspacing * xindex;
  float ev[1]={0};
  float gridspacing_u = gridspacing * BLOCKSIZEX;
  for (int a=0;a<numatoms;a++){
    float dy = coory - atominfo[a].y;
    float dyz2 = (dy*dy) + atominfo[a].z;
    float dx = coorx - atominfo[a].x;
    for(int u=0;u<1;u++){ ev[u]+= atominfo[a].w * rsqrtf(dx*dx+dyz2); dx+=gridspacing_u; }
  }
  for(int u=0;u<1;u++) energygrid[outaddr+u*BLOCKSIZEX]+=ev[u];
}
int main(){
  const int GX=8, GY=8, N=GX*BLOCKSIZEX*UNROLLX*GY*8; // generous grid
  int natoms=1;
  float4 *atom; float *grid;
  cudaMallocManaged(&atom, natoms*sizeof(float4));
  cudaMallocManaged(&grid, 100000*sizeof(float));
  for(int i=0;i<100000;i++) grid[i]=0;
  atom[0]=make_float4(2.0f, 1.5f, 0.25f/*=dz^2*/, -1.0f); // x,y,dz^2,charge
  dim3 b(BLOCKSIZEX,BLOCKSIZEX), g(GX,GY);
  cenergy<<<g,b>>>(natoms, 0.1f, grid, atom);
  cudaDeviceSynchronize();
  double s=0; for(int i=0;i<100000;i++) s+=grid[i];
  printf("sum=%.5f grid[0..7]= %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f\n",
         s,grid[0],grid[8],grid[16],grid[24],grid[32],grid[40],grid[48],grid[56]);
  return 0;
}
