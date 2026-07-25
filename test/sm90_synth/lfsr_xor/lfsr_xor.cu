#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

// Mirror of HPCC RandomAccess update inner loop: 64-bit LFSR (GF(2) poly 0x7)
// + 64-bit atomicXor into a power-of-two table. Reproduces the FAIL with
// lifted PTX (randomAccess shows ~98% wrong locations).
#define POLY 0x0000000000000007ULL

extern "C" __global__ void upd(uint64_t* __restrict__ T, const uint64_t mask, int iters, uint64_t seed) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t ran = seed + (uint64_t)j * 0x9E3779B97F4A7C15ULL; // distinct per thread
  ran |= 1;
  for (int i = 0; i < iters; i++) {
    ran = (ran << 1) ^ (((int64_t)ran < 0) ? POLY : 0);
    atomicXor((unsigned long long*)&T[ran & mask], (unsigned long long)ran);
  }
}

int main() {
  const int LOG = 12, N = 1 << LOG, THREADS = 128, ITERS = 64;
  uint64_t mask = N - 1;
  uint64_t *dT, *hT = new uint64_t[N];
  for (int i = 0; i < N; i++) hT[i] = (uint64_t)i;
  cudaMalloc(&dT, N * 8);
  cudaMemcpy(dT, hT, N * 8, cudaMemcpyHostToDevice);
  upd<<<1, THREADS>>>(dT, mask, ITERS, 0x1234567890ABCDEFULL);
  cudaDeviceSynchronize();
  cudaMemcpy(hT, dT, N * 8, cudaMemcpyDeviceToHost);
  // checksum of whole table (XOR is associative -> order-independent)
  uint64_t cksum = 0;
  for (int i = 0; i < N; i++) cksum ^= hT[i] * 0x100000001B3ULL + (uint64_t)i;
  printf("cksum=%016llx T0=%016llx T1=%016llx T2=%016llx\n",
         (unsigned long long)cksum, (unsigned long long)hT[0],
         (unsigned long long)hT[1], (unsigned long long)hT[2]);
  cudaFree(dT); delete[] hT;
  return 0;
}
