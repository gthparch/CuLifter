#include <cstdio>
#include <cuda_runtime.h>
// bool param gating a store + an int param after it (alignment), like d3q19 out_u
__global__ void k(int* out, double scale, bool flag, int n){
  int t = threadIdx.x;
  if (flag && t < n) out[t] = (int)(scale) + t + 1;
}
int main(){
  int *o; cudaMallocManaged(&o,32*4); for(int i=0;i<32;i++)o[i]=-1;
  k<<<1,32>>>(o, 100.0, true, 32); cudaDeviceSynchronize();
  int bad=0; for(int i=0;i<32;i++){ int r=100+i+1; if(o[i]!=r)bad++; }
  printf("o[0]=%d o[31]=%d bad=%d %s\n",o[0],o[31],bad,bad?"FAIL":"PASS"); return 0; }
