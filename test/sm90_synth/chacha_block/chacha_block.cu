#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

// Core ChaCha20 block transform (rotl32 funnel-shift + add) on one thread,
// plus pack4/unpack4 (PRMT). Isolates the cipher math from the hex-lookup IO.
__device__ uint32_t rotl32(uint32_t x, int n){ return (x << n) | (x >> (32 - n)); }
__device__ uint32_t pack4(const uint8_t* a){
  return uint32_t(a[0])<<0 | uint32_t(a[1])<<8 | uint32_t(a[2])<<16 | uint32_t(a[3])<<24;
}
__device__ void unpack4(uint32_t s, uint8_t* d){
  d[0]=(s>>0)&0xff; d[1]=(s>>8)&0xff; d[2]=(s>>16)&0xff; d[3]=(s>>24)&0xff;
}

extern "C" __global__ void block(const uint8_t* keybytes, uint8_t* out8){
  if (threadIdx.x != 0) return;
  uint32_t state[16];
  const char* c = "expand 32-byte k";
  state[0]=pack4((const uint8_t*)c+0); state[1]=pack4((const uint8_t*)c+4);
  state[2]=pack4((const uint8_t*)c+8); state[3]=pack4((const uint8_t*)c+12);
  for(int i=0;i<8;i++) state[4+i]=pack4(keybytes+i*4);
  state[12]=1; state[13]=0; state[14]=0x09000000; state[15]=0x4a000000;

  uint32_t r[16];
  for(int i=0;i<16;i++) r[i]=state[i];
  #define QR(x,a,b,cc,d) \
    x[a]+=x[b]; x[d]=rotl32(x[d]^x[a],16); \
    x[cc]+=x[d]; x[b]=rotl32(x[b]^x[cc],12); \
    x[a]+=x[b]; x[d]=rotl32(x[d]^x[a],8); \
    x[cc]+=x[d]; x[b]=rotl32(x[b]^x[cc],7);
  for(int i=0;i<10;i++){
    QR(r,0,4,8,12) QR(r,1,5,9,13) QR(r,2,6,10,14) QR(r,3,7,11,15)
    QR(r,0,5,10,15) QR(r,1,6,11,12) QR(r,2,7,8,13) QR(r,3,4,9,14)
  }
  for(int i=0;i<16;i++) r[i]+=state[i];
  for(int i=0;i<16;i++) unpack4(r[i], out8+i*4);
}

int main(){
  uint8_t key[32]; for(int i=0;i<32;i++) key[i]=i;
  uint8_t *dk,*dout,hout[64];
  cudaMalloc(&dk,32); cudaMalloc(&dout,64);
  cudaMemcpy(dk,key,32,cudaMemcpyHostToDevice);
  block<<<1,32>>>(dk,dout);
  cudaDeviceSynchronize();
  cudaMemcpy(hout,dout,64,cudaMemcpyDeviceToHost);
  uint64_t ck=0; for(int i=0;i<64;i++) ck=ck*131+hout[i];
  printf("cksum=%016llx b0=%02x b1=%02x b63=%02x\n",(unsigned long long)ck,hout[0],hout[1],hout[63]);
  cudaFree(dk);cudaFree(dout); return 0;
}
