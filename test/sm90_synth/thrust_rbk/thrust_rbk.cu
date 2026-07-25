#include <cstdio>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
// Minimal reduce_by_key: 64 elements, 4 segments of 16, all values=1.
// Expected each segment sum = 16. Isolates thrust decoupled-lookback scan.
int main(){
  const int N=64, S=16;
  int *h_in=new int[N], *h_keys=new int[N];
  for(int i=0;i<N;i++){h_in[i]=1; h_keys[i]=i/S;}
  int *d_in,*d_keys,*d_out,*d_kout;
  cudaMalloc(&d_in,N*4); cudaMalloc(&d_keys,N*4);
  cudaMalloc(&d_out,(N/S)*4); cudaMalloc(&d_kout,(N/S)*4);
  cudaMemcpy(d_in,h_in,N*4,cudaMemcpyHostToDevice);
  cudaMemcpy(d_keys,h_keys,N*4,cudaMemcpyHostToDevice);
  thrust::reduce_by_key(thrust::device,d_keys,d_keys+N,d_in,d_kout,d_out);
  cudaDeviceSynchronize();
  int hout[N/S]; cudaMemcpy(hout,d_out,(N/S)*4,cudaMemcpyDeviceToHost);
  for(int i=0;i<N/S;i++) printf("seg%d=%d\n",i,hout[i]);
  return 0;
}
