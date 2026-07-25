#include <cstdio>
#include <cuda_runtime.h>
#define TPB 32
// Minimal predicated shared store: only lanes with rank<16 write their rank+100;
// others must leave tmp[rank] at its init (rank). If predication is dropped,
// all lanes overwrite -> corruption visible.
__global__ void k(int* out){
  __shared__ int tmp[TPB];
  int r = threadIdx.x;
  tmp[r] = r;            // init
  __syncthreads();
  int v = tmp[r] + 100;  // would-be new value
  if (r < 16) {          // predicated store
    tmp[r] = v;
  }
  __syncthreads();
  out[r] = tmp[r];
}
int main(){
  int* dout, hout[TPB];
  cudaMalloc(&dout,TPB*4);
  k<<<1,TPB>>>(dout); cudaDeviceSynchronize();
  cudaMemcpy(hout,dout,TPB*4,cudaMemcpyDeviceToHost);
  for(int i=0;i<TPB;i++) printf("%d ",hout[i]); printf("\n");
  cudaFree(dout); return 0;
}
