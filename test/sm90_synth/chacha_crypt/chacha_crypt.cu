#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include "chacha20.h"

// Full multi-block ChaCha20 crypt path: stateful Chacha20 struct (keystream8[64]
// + position in local memory) + 64-bit counter increment across >64-byte output.
extern "C" __global__ void k(const uint8_t* key, const uint8_t* nonce, uint8_t* result, int n){
  if (threadIdx.x != 0) return;
  Chacha20 chacha(key, nonce);
  chacha.crypt(result, n);   // result is pre-zeroed -> becomes the keystream
}

int main(){
  uint8_t key[32], nonce[8];
  for(int i=0;i<32;i++) key[i]=i;
  for(int i=0;i<8;i++) nonce[i]=i;
  const int N = 256;          // 4 full blocks -> exercises counter increment
  uint8_t *dk,*dn,*dr, hr[256]={0};
  cudaMalloc(&dk,32); cudaMalloc(&dn,8); cudaMalloc(&dr,N);
  cudaMemcpy(dk,key,32,cudaMemcpyHostToDevice);
  cudaMemcpy(dn,nonce,8,cudaMemcpyHostToDevice);
  cudaMemset(dr,0,N);
  k<<<1,32>>>(dk,dn,dr,N);
  cudaDeviceSynchronize();
  cudaMemcpy(hr,dr,N,cudaMemcpyDeviceToHost);
  for(int i=0;i<N;i++) printf("%02x%s",hr[i],(i%16==15)?"\n":" ");
  cudaFree(dk);cudaFree(dn);cudaFree(dr); return 0;
}
