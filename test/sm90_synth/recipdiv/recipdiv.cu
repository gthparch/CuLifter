#include <cstdio>
#include <cuda_runtime.h>
// tensorT's index decomposition: im = it * (1.0f/shape) truncated to int,
// then remainder = it - im*shape. Must match integer it/shape, it%shape.
__global__ void k(const int* its, const int* shape, const float* recip,
                  int* im_out, int* rem_out, int N){
  int t = blockIdx.x*blockDim.x+threadIdx.x;
  if (t<N){
    int it = its[t]; int s = shape[t % 8];
    int im = it * recip[t % 8];            // int * float -> trunc to int
    int rem = it - im * s;
    im_out[t] = im; rem_out[t] = rem;
  }
}
int main(){
  const int N=200000; int shp[8]={2,3,4,5,7,16,31,64}; float rcp[8];
  for(int i=0;i<8;i++) rcp[i]=1.0f/shp[i];
  int *its,*shape,*imo,*remo; float *recip;
  cudaMallocManaged(&its,N*4); cudaMallocManaged(&shape,8*4); cudaMallocManaged(&recip,8*4);
  cudaMallocManaged(&imo,N*4); cudaMallocManaged(&remo,N*4);
  for(int i=0;i<8;i++){shape[i]=shp[i];recip[i]=rcp[i];}
  for(int i=0;i<N;i++) its[i]=i;
  k<<<(N+255)/256,256>>>(its,shape,recip,imo,remo,N); cudaDeviceSynchronize();
  int bad=0; long firstbad=-1;
  for(int t=0;t<N;t++){ int s=shp[t%8]; int im=(int)((float)its[t]*rcp[t%8]); int rem=its[t]-im*s;
    if(imo[t]!=im||remo[t]!=rem){ if(firstbad<0)firstbad=t; bad++; } }
  if(firstbad>=0) printf("bad=%d first t=%ld lifted(im=%d rem=%d) ref(im=%d rem=%d) FAIL\n",
    bad,firstbad,imo[firstbad],remo[firstbad],(int)((float)its[firstbad]*rcp[firstbad%8]),its[firstbad]-(int)((float)its[firstbad]*rcp[firstbad%8])*shp[firstbad%8]);
  else printf("bad=0 PASS\n");
  return 0;
}
