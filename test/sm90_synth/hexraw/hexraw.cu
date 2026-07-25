#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>

// Isolates ChaCha20's hex_to_raw byte-gather: src is `const char*` (signed),
// each char indexes a uint8_t lookup table, two nibbles combine into a byte.
__device__
void hex_to_raw(const char* src, const int n, uint8_t* dst, const uint8_t* char_to_uint){
  for (int i = threadIdx.x; i < n/2; i = i + blockDim.x) {
    uint8_t hi = char_to_uint[src[i*2 + 0]];
    uint8_t lo = char_to_uint[src[i*2 + 1]];
    dst[i] = (hi << 4) | lo;
  }
}

extern "C" __global__ void k(const char* text, int n, const uint8_t* c2u, uint8_t* out){
  hex_to_raw(text, n, out, c2u);
}

int main(){
  const char* h = "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f";
  int n = strlen(h);
  uint8_t c2u[256]={0};
  for(int i=0;i<10;i++) c2u[i+'0']=i;
  for(int i=0;i<26;i++) c2u[i+'a']=i+10;
  for(int i=0;i<26;i++) c2u[i+'A']=i+10;
  char *dt; uint8_t *dc,*dout, hout[128]={0};
  cudaMalloc(&dt,n+1); cudaMalloc(&dc,256); cudaMalloc(&dout,n/2);
  cudaMemcpy(dt,h,n+1,cudaMemcpyHostToDevice);
  cudaMemcpy(dc,c2u,256,cudaMemcpyHostToDevice);
  k<<<1,64>>>(dt,n,dc,dout);
  cudaDeviceSynchronize();
  cudaMemcpy(hout,dout,n/2,cudaMemcpyDeviceToHost);
  printf("b0=%02x b1=%02x b15=%02x b31=%02x\n",hout[0],hout[1],hout[15],hout[31]);
  cudaFree(dt);cudaFree(dc);cudaFree(dout); return 0;
}
