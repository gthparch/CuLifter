#include <cstdio>
#include <cuda_runtime.h>
// Isolate 128-bit vector load/store (LDG.128/STG.128) — split's uint4 keys,
// degrid's double2 complex. Permuted copy to force cross-lane vector moves.
__global__ void k(const uint4* in, uint4* out, int n){
  int i = blockIdx.x*blockDim.x+threadIdx.x;
  if (i<n){ uint4 v = in[i]; uint4 r; r.x=v.w; r.y=v.z; r.z=v.y; r.w=v.x; out[i]=r; }
}
int main(){
  const int n=1<<20; uint4 *in,*out;
  cudaMallocManaged(&in,n*16); cudaMallocManaged(&out,n*16);
  for(int i=0;i<n;i++){in[i].x=i;in[i].y=i*2+1;in[i].z=i*3+2;in[i].w=i*4+3;}
  k<<<(n+255)/256,256>>>(in,out,n); cudaDeviceSynchronize();
  int bad=0; for(int i=0;i<n;i++){ if(out[i].x!=in[i].w||out[i].y!=in[i].z||out[i].z!=in[i].y||out[i].w!=in[i].x)bad++; }
  printf("bad=%d %s\n", bad, bad==0?"PASS":"FAIL");
  return 0;
}
